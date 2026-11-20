#!/usr/bin/env python3
"""
indice_pandas.py - Construye las tablas del dataset de BCS con pandas.

Es la misma lógica de las secciones 3.3, diseño de prueba y 3.4 del notebook de preparación,
escrita como script para compararla con indice_spark.py (mismas reglas, ver reglas.py).

Entradas (en --entrada): fotos.csv, bitacora.csv y vistas.csv.
Salidas (en --salida): indice_imagenes.csv, vacas.csv y evaluaciones.csv.

Uso:
    python3 indice_pandas.py --entrada entrada --salida salida/pandas
"""
import time
T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from reglas import K, COLS_INDICE, COLS_VACAS, COLS_EVAL, MAX_EVALUACIONES  # noqa: E402

T_IMPORTS = time.perf_counter() - T0


def leer(ruta):
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def construir(entrada):
    entrada = Path(entrada)
    fotos = leer(entrada / "fotos.csv")
    bit = leer(entrada / "bitacora.csv")
    vis = leer(entrada / "vistas.csv")

    # Fotos no excluidas con su vista; la carpeta depende de la marca
    f = fotos[(fotos["excluida"] == "no") & (fotos["archivo_jpg"] != "")].copy()
    fv = f.merge(vis[["archivo_jpg", "vista", "usable", "marcado_por", "marcado_en"]], on="archivo_jpg", how="left")
    for c in ("vista", "usable", "marcado_por", "marcado_en"):
        fv[c] = fv[c].fillna("")
    fv["carpeta"] = np.where(fv["usable"] == "si", fv["vista"],
                             np.where(fv["usable"] == "no", "descartadas", "sin_marcar"))
    fv["ruta"] = "imagenes/" + fv["carpeta"] + "/" + fv["id_vaca"] + "/" + fv["archivo_jpg"].str.split("/").str[-1]

    # BCS por vaca. Las evaluaciones se ordenan por su posición en bitacora.csv
    # (primero las del Excel y después las de la hoja), no por fila_excel, que está vacía en la hoja.
    b = bit[bit["excluida"] == "no"].copy()
    b["bcs_num"] = pd.to_numeric(b["bcs"], errors="coerce")
    b = b[b["bcs_num"].notna()].copy()
    b["_pos"] = b.index
    b = b.sort_values(["id_vaca", "_pos"])
    b["orden_evaluacion"] = b.groupby("id_vaca").cumcount() + 1
    cb = b.groupby("id_vaca").agg(n_eval=("bcs_num", "size"), bcs_entrenamiento=("bcs_num", "mean"),
                                  bcs_min=("bcs_num", "min"), bcs_max=("bcs_num", "max"))
    if cb["n_eval"].max() > MAX_EVALUACIONES:
        raise ValueError(f"Hay vacas con más de {MAX_EVALUACIONES} evaluaciones; agrega otra columna bcs_n.")
    for n in range(1, MAX_EVALUACIONES + 1):
        cb[f"bcs_{n}"] = b[b["orden_evaluacion"] == n].set_index("id_vaca")["bcs_num"]
    cb["dif_evaluadores"] = (cb["bcs_max"] - cb["bcs_min"]).where(cb["n_eval"] >= 2)
    cb = cb.drop(columns=["bcs_min", "bcs_max"]).reset_index()

    # Fotos usables por vista de cada vaca
    us = fv["usable"] == "si"
    fv["_u"] = us.astype(int)
    fv["_post"] = (us & (fv["vista"] == "posterior")).astype(int)
    fv["_izq"] = (us & (fv["vista"] == "lateral_izq")).astype(int)
    fv["_der"] = (us & (fv["vista"] == "lateral_der")).astype(int)
    vac = fv.groupby("id_vaca").agg(n_fotos=("ruta", "size"), n_usables=("_u", "sum"), n_posterior=("_post", "sum"),
                                    n_lateral_izq=("_izq", "sum"), n_lateral_der=("_der", "sum")).reset_index()
    vac = vac.merge(cb, on="id_vaca", how="left")

    # Pliegues por vaca: se ordenan por BCS (y por ID) y se reparten en rotación
    eleg = vac[(vac["n_usables"] >= 1) & vac["bcs_entrenamiento"].notna()].copy()
    eleg["_idn"] = pd.to_numeric(eleg["id_vaca"], errors="coerce")
    eleg = eleg.sort_values(["bcs_entrenamiento", "_idn", "id_vaca"], na_position="last")
    eleg["fold"] = np.arange(len(eleg)) % K
    vac = vac.merge(eleg[["id_vaca", "fold"]], on="id_vaca", how="left")
    vac["fold"] = vac["fold"].astype("Int64")
    vac["en_entrenamiento"] = np.where(vac["fold"].notna(), "si", "no")
    vac["n_eval"] = vac["n_eval"].astype("Int64")

    # Índice por foto y evaluaciones
    datos_vaca = vac[["id_vaca", "fold", "en_entrenamiento", "bcs_entrenamiento", "bcs_1", "bcs_2", "bcs_3",
                      "n_eval", "dif_evaluadores"]]
    idx = fv.merge(datos_vaca, on="id_vaca", how="left")
    idx["usar_en_entrenamiento"] = np.where((idx["usable"] == "si") & (idx["en_entrenamiento"] == "si"), "si", "no")

    ev = b.merge(cb[["id_vaca", "n_eval"]], on="id_vaca", how="left")
    ev = ev.merge(vac[["id_vaca", "fold", "en_entrenamiento"]], on="id_vaca", how="left")
    ev["en_entrenamiento"] = ev["en_entrenamiento"].fillna("no")
    ev["bcs"] = ev["bcs_num"]
    ev["n_eval"] = ev["n_eval"].astype("Int64")

    def por_id(df, extra):
        df = df.assign(_idn=pd.to_numeric(df["id_vaca"], errors="coerce"))
        return df.sort_values(["_idn", "id_vaca"] + extra, na_position="last")

    idx = por_id(idx, ["ruta"])[COLS_INDICE]
    vac = por_id(vac, [])[COLS_VACAS]
    ev = ev.sort_values("_pos")[COLS_EVAL]
    return idx, vac, ev


def main():
    ap = argparse.ArgumentParser(description="Tablas del dataset BCS (pandas).")
    ap.add_argument("--entrada", default="entrada")
    ap.add_argument("--salida", default="salida/pandas")
    a = ap.parse_args()

    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    idx, vac, ev = construir(a.entrada)
    idx.to_csv(salida / "indice_imagenes.csv", index=False)
    vac.to_csv(salida / "vacas.csv", index=False)
    ev.to_csv(salida / "evaluaciones.csv", index=False)
    t_proc = time.perf_counter() - t1

    print(f"pandas: {len(idx)} fotos, {len(vac)} vacas, {len(ev)} evaluaciones -> {salida}")
    print("TIEMPOS_JSON " + json.dumps({"t_imports": T_IMPORTS, "t_sesion": 0.0, "t_procesamiento": t_proc}))


if __name__ == "__main__":
    main()
