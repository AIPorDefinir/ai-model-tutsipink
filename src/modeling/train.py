import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import config

ESQUEMA = (
    "variante string, semilla int, fold_prueba int, ruta string, id_vaca string, vista string, "
    "fold int, bcs double, pred double, info string"
)


def leer_fotos(vistas="all"):
    import pyarrow.parquet as pq

    fotos = pq.read_table(str(config.FOTOS_PARQUET), partitioning="hive").to_pandas()
    fotos["fold"] = fotos["fold"].astype(int)
    if vistas != "all":
        fotos = fotos[fotos["vista"] == vistas]
    return fotos.sort_values(["id_vaca", "vista", "ruta"]).reset_index(drop=True)


def correr_tarea(t):
    from . import network

    hp = json.loads(t["hp"])
    semilla, k = int(t["semilla"]), int(t["fold_prueba"])
    logs = config.ARTEFACTOS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    nombre = f"{hp['variante']}_s{semilla}_" + (f"k{k}" if k >= 0 else "final")
    archivo_log = open(logs / f"{nombre}.log", "w")

    def registro(msg):
        print(f"[{nombre}] {msg}", flush=True)
        archivo_log.write(msg + "\n")
        archivo_log.flush()

    t0 = time.time()
    gpus = network.configurar_gpu(hp.get("precision", "float32"))
    registro(f"GPU: {gpus}, hiperparámetros: {json.dumps(hp)}")
    fotos = leer_fotos(hp["vistas"])
    X = network.decodificar(fotos["jpg"].tolist())
    info = {"gpus": gpus}
    if k >= 0:
        kval = (k + 1) % config.K
        prueba = (fotos["fold"] == k).to_numpy()
        val = (fotos["fold"] == kval).to_numpy()
        tr = ~prueba & ~val
        f_tr, f_val = fotos[tr], fotos[val]
        modelo, mejor_epoca, historia = network.entrenar(
            hp, X[tr], f_tr["bcs"], f_tr["id_vaca"], hp["epocas_max"],
            val=(X[val], f_val["bcs"].to_numpy(), f_val["id_vaca"].to_numpy()),
            semilla=semilla + k, registro=registro,
        )
        salida = fotos[prueba].copy()
        salida["pred"] = network.predecir(modelo, X[prueba], hp)
        info.update(pliegue_val=kval, mejor_epoca=mejor_epoca,
                    mae_vaca_val=min(h["mae_vaca_val"] for h in historia))
    else:
        modelo, _, historia = network.entrenar(
            hp, X, fotos["bcs"], fotos["id_vaca"], int(t["epocas"]), semilla=semilla, registro=registro
        )
        config.ARTEFACTOS.mkdir(parents=True, exist_ok=True)
        pesos = config.ARTEFACTOS / f"tf_efficientnet_{hp['variante']}_s{semilla}.keras"
        modelo.save(pesos)
        salida = fotos.copy()
        salida["pred"] = network.predecir(modelo, X, hp)
        info.update(epocas=int(t["epocas"]), pesos=pesos.name, perdida_final=historia[-1]["perdida"])
    info.update(segundos=round(time.time() - t0), historia=historia)
    registro(f"listo: {json.dumps({k_: v for k_, v in info.items() if k_ != 'historia'})}")
    archivo_log.close()
    salida = salida.drop(columns="jpg").assign(variante=hp["variante"], semilla=semilla, fold_prueba=k, info=json.dumps(info))
    return salida[[c.split()[0] for c in ESQUEMA.split(", ")]]


def _mapear(lotes):
    for lote in lotes:
        for _, t in lote.iterrows():
            yield correr_tarea(t)


def lanzar(spark, tareas, destino):
    from pyspark.sql import Row

    filas = [Row(**t) for t in tareas]
    df = spark.sparkContext.parallelize(filas, len(filas)).toDF()
    pred = df.mapInPandas(_mapear, ESQUEMA)
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    pred.write.mode("overwrite").partitionBy("variante", "semilla").parquet(str(destino))


def validacion_cruzada(spark, hp, semillas, pliegues=range(config.K), destino="cv"):
    for s in semillas:
        t0 = time.time()
        tareas = [{"hp": json.dumps(hp), "semilla": s, "fold_prueba": k, "epocas": hp["epocas_max"]} for k in pliegues]
        print(f"CV {hp['variante']} semilla {s}: {len(tareas)} tareas", flush=True)
        lanzar(spark, tareas, config.PREDICCIONES / destino)
        print(f"CV {hp['variante']} semilla {s} lista en {time.time() - t0:.0f} s", flush=True)


def epocas_de_cv(spark, variante):
    from pyspark.sql import functions as F

    infos = (
        spark.read.parquet(str(config.PREDICCIONES / "cv"))
        .filter(F.col("variante") == variante)
        .select("semilla", "fold_prueba", "info").distinct().collect()
    )
    epocas = sorted(json.loads(r["info"])["mejor_epoca"] for r in infos)
    print(f"Mejores épocas en CV ({len(epocas)} entrenamientos): {epocas}", flush=True)
    return int(np.median(epocas))


def entrenar_final(spark, hp, semillas, epocas):
    tareas = [{"hp": json.dumps(hp), "semilla": s, "fold_prueba": -1, "epocas": epocas} for s in semillas]
    lanzar(spark, tareas, config.PREDICCIONES / "final")
    meta = {**hp, "epocas_final": epocas, "miembros": [f"tf_efficientnet_{hp['variante']}_s{s}.keras" for s in semillas]}
    (config.ARTEFACTOS / f"tf_efficientnet_{hp['variante']}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Modelo final: {len(semillas)} redes de {epocas} épocas en {config.ARTEFACTOS}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variante", required=True, choices=list(config.VARIANTES))
    p.add_argument("--cv", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--semillas", type=int, nargs="+", default=config.SEMILLAS)
    p.add_argument("--pliegues", type=int, nargs="+", default=list(range(config.K)))
    p.add_argument("--epocas-max", type=int, help="solo para pruebas cortas")
    p.add_argument("--microlote", type=int)
    p.add_argument("--precision", choices=["float32", "mixed_float16", "mixed_bfloat16"])
    a = p.parse_args()
    cambios = {k: v for k, v in {"epocas_max": a.epocas_max, "microlote": a.microlote, "precision": a.precision}.items() if v}
    hp = config.hiperparametros(a.variante, **cambios)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    spark = config.crear_spark("bcs-entrenar", una_tarea_a_la_vez=True)
    t0 = time.time()
    try:
        if a.cv:
            prueba = bool(a.epocas_max) or sorted(a.pliegues) != list(range(config.K))
            validacion_cruzada(spark, hp, a.semillas, a.pliegues, "prueba" if prueba else "cv")
        if a.final:
            entrenar_final(spark, hp, a.semillas, epocas_de_cv(spark, a.variante))
    finally:
        spark.stop()
    print(f"Listo en {time.time() - t0:.0f} s", flush=True)
