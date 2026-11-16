"""Métricas por vaca: vacas dentro de ±0.25 (n/N), MAE y % dentro de ±0.50, continuas y con salida redondeada."""
import numpy as np
import pandas as pd

from . import config


def pesos_vaca(id_vaca):
    """Peso por foto = 1 / fotos de su vaca, normalizado a media 1: cada vaca pesa lo mismo."""
    _, inv, cuenta = np.unique(np.asarray(id_vaca), return_inverse=True, return_counts=True)
    w = 1.0 / cuenta[inv]
    return w / w.mean()


def por_vaca(fotos, agregacion="media"):
    """fotos: id_vaca, bcs, pred (y opcionalmente fold). Vaca = media o mediana de sus fotos."""
    fn = {"media": "mean", "mediana": "median"}[agregacion]
    cols = {"bcs": ("bcs", "first"), "pred": ("pred", fn), "n_fotos": ("pred", "size")}
    if "fold" in fotos:
        cols["fold"] = ("fold", "first")
    return fotos.groupby("id_vaca", as_index=False).agg(**cols)


def errores_vaca(id_vaca, y, pred, agregacion="media"):
    v = por_vaca(pd.DataFrame({"id_vaca": id_vaca, "bcs": y, "pred": pred}), agregacion)
    return (v["pred"] - v["bcs"]).to_numpy()


def mae_vaca(id_vaca, y, pred, agregacion="media"):
    return float(np.abs(errores_vaca(id_vaca, y, pred, agregacion)).mean())


def dentro(errores, tolerancia=0.25):
    return int((np.abs(np.asarray(errores, float)) <= tolerancia + config.EPS).sum())


def texto(r, sufijo=""):
    return (f"{r['vacas_025' + sufijo]}/{r['n_vacas']} vacas ±0.25 ({r['pct_025' + sufijo]:.1f} %), "
            f"MAE {r['mae' + sufijo]:.3f}, ±0.50 {r['pct_050' + sufijo]:.1f} %")


def lectura_oficial(salidas, oficial):
    """Métricas de la lectura oficial de un modelo a partir de resumen["salidas"] de evaluate.evaluar."""
    sufijo = "_redondeada" if oficial.get("redondeada_025") else ""
    r = salidas[oficial["salida"]]
    return {"salida": oficial["salida"], "redondeada_025": bool(sufijo), "n_vacas": r["n_vacas"],
            **{m: r[m + sufijo] for m in ("vacas_025", "pct_025", "mae", "pct_050")}}
