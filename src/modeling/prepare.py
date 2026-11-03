import argparse
import io
import time

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType

from . import config


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


def validar(indice, vacas):
    errores = []
    por_vaca = indice.groupBy("id_vaca").agg(
        F.countDistinct("bcs").alias("etiquetas"), F.countDistinct("fold").alias("pliegues")
    )
    malas = por_vaca.filter("etiquetas != 1 OR pliegues != 1").count()
    if malas:
        errores.append(f"{malas} vacas con más de una etiqueta o pliegue")
    vistas_malas = indice.filter(~F.col("vista").isin(config.VISTAS)).count()
    if vistas_malas:
        errores.append(f"{vistas_malas} fotos con vista inválida")
    fuera = indice.filter((F.col("bcs") < config.BCS_MIN) | (F.col("bcs") > config.BCS_MAX)).count()
    if fuera:
        errores.append(f"{fuera} fotos con BCS fuera de escala")
    pliegues = sorted(r["fold"] for r in indice.select("fold").distinct().collect())
    if pliegues != list(range(config.K)):
        errores.append(f"pliegues {pliegues}, se esperaban 0 a {config.K - 1}")
    v_entr = vacas.filter("en_entrenamiento = 'si'").select("id_vaca")
    dif = v_entr.exceptAll(indice.select("id_vaca").distinct()).count() + indice.select(
        "id_vaca"
    ).distinct().exceptAll(v_entr).count()
    if dif:
        errores.append(f"{dif} vacas no coinciden entre indice_imagenes.csv y vacas.csv")
    if errores:
        raise ValueError("Validación del índice: " + "; ".join(errores))


def preparar(spark, base):
    t0 = time.time()
    indice = (
        leer_csv(spark, base, "indice_imagenes.csv")
        .filter("usar_en_entrenamiento = 'si'")
        .select(
            "ruta",
            "id_vaca",
            "vista",
            F.col("fold").cast("int").alias("fold"),
            F.col("bcs_entrenamiento").cast("double").alias("bcs"),
        )
    )
    evaluaciones = (
        leer_csv(spark, base, "evaluaciones.csv")
        .filter("en_entrenamiento = 'si'")
        .select("id_vaca", F.col("bcs").cast("double").alias("bcs"))
    )
    vacas = leer_csv(spark, base, "vacas.csv")
    indice = indice.cache()
    validar(indice, vacas)
    n_fotos, n_vacas = indice.count(), indice.select("id_vaca").distinct().count()
    print(f"Índice: {n_fotos} fotos de {n_vacas} vacas, validación correcta ({time.time() - t0:.0f} s)", flush=True)

    rutas = [f"{base}/imagenes/{v}" for v in config.VISTAS]
    binarios = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(rutas)
        .select(
            F.regexp_extract("path", r"(imagenes/.+)$", 1).alias("ruta"),
            F.col("content").alias("original"),
        )
    )
    fotos = (
        indice.join(binarios, "ruta", "left")
        .withColumn("jpg", F.when(F.col("original").isNotNull(), reducir_udf("original")))
        .drop("original")
        .repartition(config.K, "fold")
    )
    fotos.write.mode("overwrite").partitionBy("fold").parquet(str(config.FOTOS_PARQUET))
    guardadas = spark.read.parquet(str(config.FOTOS_PARQUET))
    faltan = guardadas.filter("jpg IS NULL").count()
    if faltan:
        raise FileNotFoundError(f"{faltan} fotos del índice no están en {base}/imagenes/")
    if guardadas.count() != n_fotos:
        raise ValueError("El Parquet de fotos no tiene el mismo número de filas que el índice")
    evaluaciones.write.mode("overwrite").parquet(str(config.EVALUACIONES_PARQUET))
    vacas.write.mode("overwrite").parquet(str(config.VACAS_PARQUET))

    resumen = guardadas.groupBy("fold").agg(
        F.countDistinct("id_vaca").alias("vacas"),
        F.count("*").alias("fotos"),
        F.round(F.avg("bcs"), 3).alias("bcs_medio"),
        F.round(F.sum(F.length("jpg")) / 1e6, 1).alias("MB"),
    )
    resumen.orderBy("fold").show()
    guardadas.groupBy("vista").count().orderBy("vista").show()
    print(f"Preparación lista en {time.time() - t0:.0f} s: {config.SPARK_DATA}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local", default="data/dataset_bcs", help="carpeta de dataset_bcs/ (relativa al repo)")
    p.add_argument("--s3", action="store_true", help="leer directo del bucket con s3a en lugar de la copia local")
    a = p.parse_args()
    spark = config.crear_spark("bcs-preparar", s3=a.s3)
    try:
        base = config.ruta_s3a() if a.s3 else str((config.RAIZ / a.local).resolve())
        preparar(spark, base)
    finally:
        spark.stop()
