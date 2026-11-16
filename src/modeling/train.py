"""Entrenamiento por pliegue del Modelo 1: época de paro en un pliegue interno y reentrenamiento con todo el TRAIN."""
import json
import time

import numpy as np
import pandas as pd

from . import config, protocol


def leer_fotos(entrada="original"):
    import pyarrow.parquet as pq

    fotos = pq.read_table(str(config.fotos_parquet(entrada)), partitioning="hive").to_pandas()
    fotos["fold"] = fotos["fold"].astype(int)
    fotos["id_vaca"] = fotos["id_vaca"].astype(str)
    return fotos.sort_values(["id_vaca", "vista", "ruta"]).reset_index(drop=True)


def tarea_cnn(cfg, k):
    from . import network
    from .metrics import por_vaca
    from .tasks import filas_vacas

    hp = config.hiperparametros_cnn(**cfg.get("hiperparametros", {}))
    protocolo = cfg["protocolo"]
    semilla = int(cfg["semilla"]) + k
    config.REGISTROS.mkdir(parents=True, exist_ok=True)
    nombre = f"{cfg['nombre']}_k{k}"
    with open(config.REGISTROS / f"{nombre}.log", "w") as archivo_log:
        def registro(msg):
            print(f"[{nombre}] {msg}", flush=True)
            archivo_log.write(msg + "\n")
            archivo_log.flush()

        t0 = time.time()
        gpus = network.configurar_gpu(hp["dtype"])
        registro(f"GPU: {gpus}, hiperparámetros: {json.dumps(hp)}")
        fotos = leer_fotos(cfg["entrada"])
        X = network.decodificar(fotos["jpg"].tolist())
        fold, vacas, y = fotos["fold"].to_numpy(), fotos["id_vaca"].to_numpy(), fotos["bcs"].to_numpy(float)
        ev, tr = protocol.particion(fold, k, protocolo)
        protocol.revisar(vacas, ev, tr)
        paro = protocol.pliegue_paro(k, protocolo)
        ival = tr & (fold == paro)
        itr = tr & ~ival
        _, mejor_epoca, historia = network.entrenar(
            hp, X[itr], y[itr], vacas[itr], hp["epocas_max"], val=(X[ival], y[ival], vacas[ival]),
            semilla=semilla, registro=registro)
        mejor_epoca = max(1, int(mejor_epoca))
        modelo, _, _ = network.entrenar(hp, X[tr], y[tr], vacas[tr], mejor_epoca, semilla=semilla, registro=registro)
        pred = network.predecir(modelo, X[ev], hp)
        v = por_vaca(pd.DataFrame({"id_vaca": vacas[ev], "bcs": y[ev], "pred": pred}), "media")
        info = {"pliegue_paro": paro, "mejor_epoca": mejor_epoca, "gpus": gpus,
                "mae_vaca_paro": min(h["mae_vaca_val"] for h in historia), "semilla": semilla,
                "segundos": round(time.time() - t0)}
        registro(f"listo: {json.dumps(info)}")
    return filas_vacas(v, cfg["nombre"], "continua", k, protocol.pliegue_evaluado(k, protocolo), y[tr], info)

