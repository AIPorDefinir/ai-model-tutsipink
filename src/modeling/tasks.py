"""Tareas por pliegue externo que Spark reparte con mapInPandas; cada una devuelve predicciones por vaca."""
import json
import time

import numpy as np
import pandas as pd

from . import protocol

ESQUEMA_VACAS = ("modelo string, salida string, k int, id_vaca string, fold int, bcs double, pred double, "
                 "bcs_min_train double, bcs_max_train double, info string")
COLUMNAS_VACAS = [c.split()[0] for c in ESQUEMA_VACAS.split(", ")]


def filas_vacas(vacas, modelo, salida, k, fold_ev, y_tr, info):
    """vacas: DataFrame id_vaca, bcs, pred (una fila por vaca evaluada)."""
    return pd.DataFrame({
        "modelo": modelo, "salida": salida, "k": k, "id_vaca": vacas["id_vaca"].astype(str).to_numpy(),
        "fold": fold_ev, "bcs": vacas["bcs"].to_numpy(float), "pred": vacas["pred"].to_numpy(float),
        "bcs_min_train": float(np.min(y_tr)), "bcs_max_train": float(np.max(y_tr)),
        "info": json.dumps(info, default=float, ensure_ascii=False),
    })[COLUMNAS_VACAS]


def _datos_embeddings(cfg, k):
    from . import embeddings

    E, ix = embeddings.cargar(cfg["extractor"], cfg["entrada"])
    y, vacas, fold = ix["bcs"].to_numpy(float), ix["id_vaca"].to_numpy(), ix["fold"].to_numpy(int)
    ev, tr = protocol.particion(fold, k, cfg["protocolo"])
    protocol.revisar(vacas, ev, tr)
    return E, y, vacas, fold, ev, tr


def tarea_regresion(cfg, k):
    """Modelos 2 y 4: regresión sobre embeddings congelados con receta elegida por CV anidada."""
    from . import heads
    from .metrics import por_vaca

    t0 = time.time()
    E, y, vacas, fold, ev, tr = _datos_embeddings(cfg, k)
    cab = cfg.get("cabeza", {})
    pred, receta, info, puntajes = heads.seleccion_anidada(
        E, y, vacas, fold, tr, ev, cab.get("bases", list(heads.BASES)), cab.get("estiramientos", list(heads.ESTIRAMIENTOS)),
        cfg["agregacion_vaca"])
    v = por_vaca(pd.DataFrame({"id_vaca": vacas[ev], "bcs": y[ev], "pred": pred}), cfg["agregacion_vaca"])
    info = {"receta": f"{receta[0]}|{receta[1]}", **info, "pliegues_train": sorted(set(fold[tr].tolist())),
            "segundos": round(time.time() - t0, 1)}
    return filas_vacas(v, cfg["nombre"], "continua", k, protocol.pliegue_evaluado(k, cfg["protocolo"]), y[tr], info)


def tarea_clasificacion(cfg, k):
    """Modelo 5: logística multiclase en la rejilla de 0.25 (valor esperado y clase más probable)."""
    from . import classification

    t0 = time.time()
    E, y, vacas, fold, ev, tr = _datos_embeddings(cfg, k)
    salidas, info = classification.seleccion_anidada(E, y, vacas, fold, tr, ev)
    info["segundos"] = round(time.time() - t0, 1)
    fold_ev = protocol.pliegue_evaluado(k, cfg["protocolo"])
    return pd.concat([filas_vacas(v, cfg["nombre"], dec, k, fold_ev, y[tr], info) for dec, v in salidas.items()],
                     ignore_index=True)


def correr(t):
    cfg, k = json.loads(t["cfg"]), int(t["k"])
    tipo = cfg["tipo"]
    if tipo == "embeddings_regresion":
        return tarea_regresion(cfg, k)
    if tipo == "embeddings_clasificacion":
        return tarea_clasificacion(cfg, k)
    if tipo == "cnn":
        from .train import tarea_cnn

        return tarea_cnn(cfg, k)
    if tipo == "ajuste_fino":
        from .finetune import tarea_miembro

        return tarea_miembro(cfg, k, int(t["semilla"]))
    raise ValueError(f"tipo de modelo desconocido: {tipo}")


def _mapear(lotes):
    for lote in lotes:
        for _, t in lote.iterrows():
            yield correr(t)


def lanzar(spark, tareas, destino):
    """tareas: lista de dicts con cfg (JSON), k y semilla. Escribe las predicciones por vaca en `destino`."""
    from pyspark.sql import Row

    filas = [Row(**t) for t in tareas]
    df = spark.sparkContext.parallelize(filas, len(filas)).toDF()
    pred = df.mapInPandas(_mapear, ESQUEMA_VACAS)
    pred.write.mode("overwrite").partitionBy("salida").parquet(str(destino))
    return destino
