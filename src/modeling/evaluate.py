"""Evaluación por vaca con Spark SQL; guarda solo agregados en $BCS_RESULTADOS/<modelo>/resumen.json.

Uso: python -m src.modeling.evaluate --modelo resnet50_vacas
"""
import argparse
import json

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from . import config
from .metrics import lectura_oficial, texto


def con_redondeo(vacas: DataFrame) -> DataFrame:
    """Salida redondeada a 0.25: múltiplo de 0.25 más cercano, recortado al rango de BCS del TRAIN."""
    r = F.floor(F.col("pred") / config.PASO_BCS + 0.5) * config.PASO_BCS
    return vacas.withColumn("pred_redondeada", F.least(F.greatest(r, F.col("bcs_min_train")), F.col("bcs_max_train")))


def _agregados(col_pred, sufijo):
    err = F.abs(F.col(col_pred) - F.col("bcs"))
    return [
        F.count(F.lit(1)).alias("n_vacas"),
        F.sum(F.when(err <= 0.25 + config.EPS, 1).otherwise(0)).alias(f"vacas_025{sufijo}"),
        F.avg(err).alias(f"mae{sufijo}"),
        F.avg(F.when(err <= 0.5 + config.EPS, 100.0).otherwise(0.0)).alias(f"pct_050{sufijo}"),
    ]


def metricas(vacas: DataFrame, por=("modelo", "salida")) -> DataFrame:
    v = con_redondeo(vacas)
    cont = v.groupBy(*por).agg(*_agregados("pred", ""))
    red = v.groupBy(*por).agg(*_agregados("pred_redondeada", "_redondeada")).drop("n_vacas")
    m = cont.join(red, list(por))
    return (m.withColumn("pct_025", 100.0 * F.col("vacas_025") / F.col("n_vacas"))
             .withColumn("pct_025_redondeada", 100.0 * F.col("vacas_025_redondeada") / F.col("n_vacas")))


def revisar_cobertura(vacas: DataFrame):
    """Cada vaca debe evaluarse exactamente una vez por salida."""
    dup = vacas.groupBy("salida", "id_vaca").count().filter("count != 1").count()
    if dup:
        raise ValueError(f"{dup} vacas evaluadas más de una vez en alguna salida")


def evaluar(spark, modelo, salidas=None, oficial=None):
    vacas = spark.read.parquet(str(config.PREDICCIONES / modelo))
    if salidas:
        vacas = vacas.filter(F.col("salida").isin(*salidas))
    vacas = vacas.cache()
    revisar_cobertura(vacas)
    total = metricas(vacas).toPandas()
    pliegues = metricas(vacas, ("modelo", "salida", "fold")).toPandas().sort_values(["salida", "fold"])
    elecciones = vacas.select("salida", "k", "info").distinct().toPandas().sort_values(["salida", "k"])
    carpeta = config.RESULTADOS / modelo
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen = {
        "modelo": modelo,
        "titulo": config.MODELOS.get(modelo, modelo),
        "metrica_principal": "vacas con |error| <= 0.25 (n/N), por vaca",
        "salidas": {r["salida"]: {k: (None if v != v else v) for k, v in r.drop(["modelo", "salida"]).to_dict().items()}
                    for _, r in total.iterrows()},
        "por_pliegue": pliegues.drop(columns="modelo").to_dict("records"),
        "elecciones_por_pliegue": [{"salida": r["salida"], "k": int(r["k"]), **json.loads(r["info"])}
                                   for _, r in elecciones.iterrows()],
    }
    if oficial:
        resumen["lectura_oficial"] = lectura_oficial(resumen["salidas"], oficial)
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=float))
    for _, r in total.sort_values("salida").iterrows():
        print(f"{config.MODELOS.get(modelo, modelo)} [{r['salida']}]", flush=True)
        print(f"  continua:               {texto(r)}", flush=True)
        print(f"  salida redondeada 0.25: {texto(r, '_redondeada')}", flush=True)
    if oficial:
        o = resumen["lectura_oficial"]
        lectura = o["salida"] + (", salida redondeada a 0.25" if o["redondeada_025"] else ", continua")
        print(f"Lectura oficial ({lectura}): {texto(o)}", flush=True)
    print(f"Resumen agregado en {carpeta / 'resumen.json'}", flush=True)
    return resumen


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", required=True, help="identificador del modelo (nombre de su configuración)")
    a = p.parse_args()
    try:
        cfg = config.cargar_config(a.modelo)
    except FileNotFoundError:
        cfg = {}
    spark = config.crear_spark("bcs-evaluar")
    try:
        evaluar(spark, a.modelo, cfg.get("salidas"), cfg.get("lectura_oficial"))
    finally:
        spark.stop()
