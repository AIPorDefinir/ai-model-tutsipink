#!/usr/bin/env python3
"""
indice_spark.py - Construye las tablas del dataset de BCS con PySpark.

Mismas reglas que indice_pandas.py (ver reglas.py). Corre en modo local o en un clúster de Spark:

    python3 indice_spark.py --entrada entrada --salida salida/spark
    python3 indice_spark.py --entrada entrada --salida salida/spark --master spark://<ip-del-master>:7077

En modo local, Spark lee y escribe los CSV directamente. En un clúster, los workers no ven los archivos
del driver, así que el driver lee los tres CSV de entrada, los reparte al clúster como DataFrames y al final
recibe las tablas (que son chicas) para escribirlas. El cálculo (uniones, agregaciones, ventanas y pliegues)
corre en Spark en los dos casos.
"""
import time
T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

from pyspark.sql import SparkSession, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import StringType, StructField, StructType  # noqa: E402

from reglas import K, COLS_INDICE, COLS_VACAS, COLS_EVAL, MAX_EVALUACIONES  # noqa: E402

T_IMPORTS = time.perf_counter() - T0


def es_local(master):
    return master.startswith("local")


def leer(spark, ruta, local):
    """Lee un CSV con todo como texto, con los vacíos como "" (igual que pandas) y su posición en el archivo."""
    if local:
        df = (spark.read.option("header", True).option("escape", '"').option("encoding", "UTF-8")
              .csv(str(Path(ruta).resolve())))
        for c in df.columns:
            limpio = c.replace("﻿", "").strip()
            if limpio != c:
                df = df.withColumnRenamed(c, limpio)
        # Con un solo archivo, monotonically_increasing_id crece en el orden de las filas del archivo
        return df.fillna("").withColumn("_pos", F.monotonically_increasing_id())
    import pandas as pd
    pdf = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    pdf.columns = [str(c).replace("﻿", "").strip() for c in pdf.columns]
    pdf["_pos"] = range(len(pdf))
    pdf["_pos"] = pdf["_pos"].astype(str)
    esquema = StructType([StructField(c, StringType(), True) for c in pdf.columns])
    df = spark.createDataFrame(pdf, schema=esquema)
    return df.withColumn("_pos", F.col("_pos").cast("long"))


def construir(spark, entrada, local):
    entrada = Path(entrada)
    fotos = leer(spark, entrada / "fotos.csv", local)
    bit = leer(spark, entrada / "bitacora.csv", local)
    vis = leer(spark, entrada / "vistas.csv", local)

    # Fotos no excluidas con su vista; la carpeta depende de la marca
    f = fotos.filter((F.col("excluida") == "no") & (F.col("archivo_jpg") != "")).drop("_pos")
    v = vis.select("archivo_jpg", "vista", "usable", "marcado_por", "marcado_en")
    fv = (f.join(v, on="archivo_jpg", how="left")
           .fillna("", subset=["vista", "usable", "marcado_por", "marcado_en"]))
    carpeta = (F.when(F.col("usable") == "si", F.col("vista"))
                .when(F.col("usable") == "no", F.lit("descartadas"))
                .otherwise(F.lit("sin_marcar")))
    base = F.element_at(F.split(F.col("archivo_jpg"), "/"), -1)
    fv = (fv.withColumn("carpeta", carpeta)
            .withColumn("ruta", F.concat_ws("/", F.lit("imagenes"), F.col("carpeta"), F.col("id_vaca"), base)))

    # BCS por vaca. Las evaluaciones se ordenan por su posición en bitacora.csv
    # (primero las del Excel y después las de la hoja), no por fila_excel, que está vacía en la hoja.
    b = (bit.filter(F.col("excluida") == "no")
            .withColumn("bcs_num", F.col("bcs").cast("double"))
            .filter(F.col("bcs_num").isNotNull()))
    w_eval = Window.partitionBy("id_vaca").orderBy(F.col("_pos").asc())
    b = b.withColumn("orden_evaluacion", F.row_number().over(w_eval))
    aggs = [F.count(F.lit(1)).alias("n_eval"), F.avg("bcs_num").alias("bcs_entrenamiento")]
    aggs += [F.max(F.when(F.col("orden_evaluacion") == n, F.col("bcs_num"))).alias(f"bcs_{n}")
             for n in range(1, MAX_EVALUACIONES + 1)]
    aggs.append(F.when(F.count(F.lit(1)) >= 2, F.max("bcs_num") - F.min("bcs_num")).alias("dif_evaluadores"))
    cb = b.groupBy("id_vaca").agg(*aggs)
    max_eval = cb.agg(F.max("n_eval")).collect()[0][0] or 0
    if max_eval > MAX_EVALUACIONES:
        raise ValueError(f"Hay vacas con más de {MAX_EVALUACIONES} evaluaciones; agrega otra columna bcs_n.")

    # Fotos usables por vista de cada vaca
    us = F.col("usable") == "si"
    vac = fv.groupBy("id_vaca").agg(
        F.count(F.lit(1)).alias("n_fotos"),
        F.sum(us.cast("int")).alias("n_usables"),
        F.sum((us & (F.col("vista") == "posterior")).cast("int")).alias("n_posterior"),
        F.sum((us & (F.col("vista") == "lateral_izq")).cast("int")).alias("n_lateral_izq"),
        F.sum((us & (F.col("vista") == "lateral_der")).cast("int")).alias("n_lateral_der"),
    ).join(cb, on="id_vaca", how="left")

    # Pliegues por vaca: se ordenan por BCS (y por ID) y se reparten en rotación
    w_fold = Window.orderBy(F.col("bcs_entrenamiento").asc(), F.col("id_vaca").cast("long").asc_nulls_last(),
                            F.col("id_vaca").asc())
    eleg = (vac.filter((F.col("n_usables") >= 1) & F.col("bcs_entrenamiento").isNotNull())
               .withColumn("fold", ((F.row_number().over(w_fold) - 1) % K).cast("int"))
               .select("id_vaca", "fold"))
    vac = (vac.join(eleg, on="id_vaca", how="left")
              .withColumn("en_entrenamiento",
                          F.when(F.col("fold").isNotNull(), F.lit("si")).otherwise(F.lit("no"))))

    # Índice por foto y evaluaciones
    datos_vaca = vac.select("id_vaca", "fold", "en_entrenamiento", "bcs_entrenamiento", "bcs_1", "bcs_2", "bcs_3",
                            "n_eval", "dif_evaluadores")
    idx = (fv.join(datos_vaca, on="id_vaca", how="left")
             .withColumn("usar_en_entrenamiento",
                         F.when(us & (F.col("en_entrenamiento") == "si"), F.lit("si")).otherwise(F.lit("no"))))
    ev = (b.join(cb.select("id_vaca", "n_eval"), on="id_vaca", how="left")
           .join(vac.select("id_vaca", "fold", "en_entrenamiento"), on="id_vaca", how="left")
           .fillna("no", subset=["en_entrenamiento"])
           .withColumn("bcs", F.col("bcs_num")))

    idn = F.col("id_vaca").cast("long").asc_nulls_last()
    idx = idx.orderBy(idn, F.col("id_vaca").asc(), F.col("ruta").asc()).select(*COLS_INDICE)
    vac = vac.orderBy(idn, F.col("id_vaca").asc()).select(*COLS_VACAS)
    ev = ev.orderBy(F.col("_pos").asc()).select(*COLS_EVAL)
    return idx, vac, ev


def escribir(df, ruta_csv, local):
    """Escribe un solo CSV. En modo local lo escribe Spark; en un clúster el driver recibe la tabla y la escribe."""
    ruta_csv = Path(ruta_csv).resolve()
    if not local:
        df.toPandas().to_csv(ruta_csv, index=False)
        return
    tmp = ruta_csv.parent / (ruta_csv.stem + "_spark_tmp")
    df.coalesce(1).write.mode("overwrite").option("header", True).option("escape", '"').csv(str(tmp))
    parte = next(tmp.glob("part-*.csv"))
    shutil.move(str(parte), str(ruta_csv))
    shutil.rmtree(tmp)


def main():
    ap = argparse.ArgumentParser(description="Tablas del dataset BCS (PySpark).")
    ap.add_argument("--entrada", default="entrada")
    ap.add_argument("--salida", default="salida/spark")
    ap.add_argument("--master", default="local[*]", help="local[*] o spark://<ip-del-master>:7077")
    a = ap.parse_args()
    local = es_local(a.master)

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    t1 = time.perf_counter()
    spark = (SparkSession.builder.master(a.master).appName("indice_bcs")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    t_sesion = time.perf_counter() - t1

    t2 = time.perf_counter()
    idx, vac, ev = construir(spark, a.entrada, local)
    escribir(idx, salida / "indice_imagenes.csv", local)
    escribir(vac, salida / "vacas.csv", local)
    escribir(ev, salida / "evaluaciones.csv", local)
    t_proc = time.perf_counter() - t2

    ejecutores = spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()
    spark.stop()
    print(f"spark ({a.master}, {ejecutores} procesos de Spark incluido el driver) -> {salida}")
    print("TIEMPOS_JSON " + json.dumps({"t_imports": T_IMPORTS, "t_sesion": t_sesion, "t_procesamiento": t_proc,
                                        "master": a.master, "procesos_spark": ejecutores}))


if __name__ == "__main__":
    main()
