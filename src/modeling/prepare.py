"""Ingesta, validación, filtrado de fotos usables, unión foto-vaca y partición por pliegue con PySpark.

Uso: python -m src.modeling.prepare [--s3] [--local CARPETA]
"""
import argparse
import io
import os
import time

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType

from . import config

C = config.COLUMNAS


def reducir_jpg(contenido, lado=config.LADO_FOTO):
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(contenido)) as im:
        im.draft("RGB", (lado, lado))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((lado, lado), Image.Resampling.LANCZOS)
        salida = io.BytesIO()
        im.save(salida, "JPEG", quality=95)
        return salida.getvalue()


@pandas_udf(BinaryType())
def reducir_udf(contenido: pd.Series) -> pd.Series:
    return contenido.map(reducir_jpg)


def leer_csv(spark, base, nombre):
    return spark.read.csv(f"{base}/{nombre}", header=True, inferSchema=False)


def _faltan(df, columnas, archivo):
    faltan = [c for c in columnas if c not in df.columns]
    return [f"{archivo}: faltan columnas {faltan}"] if faltan else []


def unir_fotos_vacas(indice_csv, vacas_csv):
    """Fotos usables de vacas elegibles, con fold y BCS tomados de vacas.csv. Devuelve (fotos, vacas, errores)."""
    errores = _faltan(indice_csv, [C["ruta"], C["id_vaca"], C["vista"], C["usar_foto"], C["fold"], C["bcs"]],
                      "indice_imagenes.csv")
    errores += _faltan(vacas_csv, [C["id_vaca"], C["usar_vaca"], C["fold"], C["bcs"]], "vacas.csv")
    if errores:
        raise ValueError("Validación del índice: " + "; ".join(errores))
    fotos = indice_csv.filter(F.col(C["usar_foto"]) == "si").select(
        F.col(C["ruta"]).alias("ruta"), F.trim(F.col(C["id_vaca"])).alias("id_vaca"), F.col(C["vista"]).alias("vista"),
        F.col(C["fold"]).cast("int").alias("fold_indice"), F.col(C["bcs"]).cast("double").alias("bcs_indice"))
    vacas = vacas_csv.filter(F.col(C["usar_vaca"]) == "si").select(
        F.trim(F.col(C["id_vaca"])).alias("id_vaca"), F.col(C["fold"]).cast("int").alias("fold"),
        F.col(C["bcs"]).cast("double").alias("bcs"))
    unidas = fotos.join(vacas, "id_vaca", "inner")
    errores = []
    distintas = unidas.filter((F.col("fold") != F.col("fold_indice")) | (F.abs(F.col("bcs") - F.col("bcs_indice")) > 1e-9))
    n = distintas.count()
    if n:
        errores.append(f"{n} fotos con pliegue o BCS distinto entre indice_imagenes.csv y vacas.csv")
    sin_fotos = vacas.join(fotos.select("id_vaca").distinct(), "id_vaca", "left_anti").count()
    if sin_fotos:
        errores.append(f"{sin_fotos} vacas elegibles sin fotos usables en el índice")
    huerfanas = fotos.select("id_vaca").distinct().join(vacas, "id_vaca", "left_anti").count()
    if huerfanas:
        errores.append(f"{huerfanas} vacas con fotos usables que no son elegibles en vacas.csv")
    return unidas.drop("fold_indice", "bcs_indice"), vacas, errores


def validar(fotos, vacas, errores_previos=()):
    errores = list(errores_previos)
    por_vaca = fotos.groupBy("id_vaca").agg(F.countDistinct("bcs").alias("etiquetas"), F.countDistinct("fold").alias("pliegues"))
    malas = por_vaca.filter("etiquetas != 1 OR pliegues != 1").count()
    if malas:
        errores.append(f"{malas} vacas con más de una etiqueta o pliegue")
    vistas_malas = fotos.filter(~F.col("vista").isin(config.VISTAS)).count()
    if vistas_malas:
        errores.append(f"{vistas_malas} fotos con vista inválida")
    fuera = fotos.filter(F.col("bcs").isNull() | (F.col("bcs") < config.BCS_MIN) | (F.col("bcs") > config.BCS_MAX)).count()
    if fuera:
        errores.append(f"{fuera} fotos con BCS vacío o fuera de escala")
    pliegues = sorted(r["fold"] for r in vacas.select("fold").distinct().collect())
    if pliegues != list(range(config.K)):
        errores.append(f"pliegues {pliegues}, se esperaban 0 a {config.K - 1}")
    duplicadas = fotos.groupBy("ruta").count().filter("count > 1").count()
    if duplicadas:
        errores.append(f"{duplicadas} rutas de foto duplicadas")
    esperadas = os.environ.get("BCS_VACAS_ESPERADAS")
    n_vacas = fotos.select("id_vaca").distinct().count()
    if esperadas and n_vacas != int(esperadas):
        errores.append(f"{n_vacas} vacas, se esperaban {esperadas} (BCS_VACAS_ESPERADAS)")
    if errores:
        raise ValueError("Validación del índice: " + "; ".join(errores))
    return n_vacas


def preparar(spark, base, destino=None):
    t0 = time.time()
    destino = destino or config.FOTOS_PARQUET
    fotos, vacas, errores = unir_fotos_vacas(leer_csv(spark, base, "indice_imagenes.csv"),
                                             leer_csv(spark, base, "vacas.csv"))
    fotos = fotos.cache()
    n_vacas = validar(fotos, vacas, errores)
    n_fotos = fotos.count()
    print(f"Índice: {n_fotos} fotos usables de {n_vacas} vacas, validación correcta ({time.time() - t0:.0f} s)", flush=True)

    binarios = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(f"{base}/imagenes")
        .select(F.regexp_extract("path", r"(imagenes/.+)$", 1).alias("ruta"), F.col("content").alias("original"))
    )
    salida = (
        fotos.join(binarios, "ruta", "left")
        .withColumn("jpg", F.when(F.col("original").isNotNull(), reducir_udf("original")))
        .drop("original")
        .repartition(config.K, "fold")
    )
    salida.write.mode("overwrite").partitionBy("fold").parquet(str(destino))
    guardadas = spark.read.parquet(str(destino))
    faltan = guardadas.filter("jpg IS NULL").count()
    if faltan:
        raise FileNotFoundError(f"{faltan} fotos del índice no están en {base}/imagenes/")
    if guardadas.count() != n_fotos:
        raise ValueError("El Parquet de fotos no tiene el mismo número de filas que el índice")
    vacas.write.mode("overwrite").parquet(str(config.VACAS_PARQUET))

    guardadas.groupBy("fold").agg(
        F.countDistinct("id_vaca").alias("vacas"),
        F.count("*").alias("fotos"),
        F.round(F.avg("bcs"), 3).alias("bcs_medio_fotos"),
    ).orderBy("fold").show()
    guardadas.groupBy("vista").count().orderBy("vista").show()
    print(f"Preparación lista en {time.time() - t0:.0f} s: {destino}", flush=True)
    return destino


def base_datos(s3=False, local=None):
    return config.ruta_s3a() if s3 else str((local or config.DATASET_LOCAL).resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local", help="carpeta con vacas.csv, indice_imagenes.csv e imagenes/ (por omisión $BCS_DATASET_LOCAL)")
    p.add_argument("--s3", action="store_true", help="leer directo del bucket con s3a en lugar de la copia local")
    a = p.parse_args()
    from pathlib import Path

    spark = config.crear_spark("bcs-preparar", s3=a.s3)
    try:
        preparar(spark, base_datos(a.s3, Path(a.local) if a.local else None))
    finally:
        spark.stop()
