import json

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from . import config

EPS = 1e-9
METRICAS = ["mae", "rmse", "dentro_025", "dentro_050"]


def _agg(err, prefijo):
    return [
        F.avg(err).alias(f"{prefijo}_mae"),
        F.sqrt(F.avg(err * err)).alias(f"{prefijo}_rmse"),
        F.avg(F.when(err <= 0.25 + EPS, 100.0).otherwise(0.0)).alias(f"{prefijo}_dentro_025"),
        F.avg(F.when(err <= 0.5 + EPS, 100.0).otherwise(0.0)).alias(f"{prefijo}_dentro_050"),
        F.count(F.lit(1)).alias(f"{prefijo}_n"),
    ]


def por_vaca(fotos: DataFrame) -> DataFrame:
    return fotos.groupBy("modelo", "id_vaca").agg(
        F.first("fold").alias("fold"), F.first("bcs").alias("bcs"),
        F.avg("pred").alias("pred"), F.count(F.lit(1)).alias("n_fotos"),
    )


def metricas_por_pliegue(fotos: DataFrame, evaluaciones: DataFrame) -> DataFrame:
    err = F.abs(F.col("pred") - F.col("bcs"))
    vacas = por_vaca(fotos)
    m_foto = fotos.groupBy("modelo", "fold").agg(*_agg(err, "foto"))
    m_vaca = vacas.groupBy("modelo", "fold").agg(*_agg(err, "vaca"))
    m_eval = (
        vacas.select("modelo", "id_vaca", "fold", "pred")
        .join(evaluaciones.withColumnRenamed("bcs", "bcs_eval"), "id_vaca")
        .groupBy("modelo", "fold")
        .agg(F.avg(F.abs(F.col("pred") - F.col("bcs_eval"))).alias("vaca_mae_eval"))
    )
    return m_foto.join(m_vaca, ["modelo", "fold"]).join(m_eval, ["modelo", "fold"], "left")


def resumen(por_pliegue: DataFrame) -> DataFrame:
    cols = [f"{n}_{m}" for n in ("foto", "vaca") for m in METRICAS] + ["vaca_mae_eval"]
    aggs = []
    for c in cols:
        aggs += [F.avg(c).alias(f"{c}_media"), F.stddev_samp(c).alias(f"{c}_std")]
    aggs += [F.sum("foto_n").alias("foto_n"), F.sum("vaca_n").alias("vaca_n")]
    return por_pliegue.groupBy("modelo").agg(*aggs)


def nombre_modelo(variante, semilla):
    base = f"tf_efficientnet_{variante}"
    return base if semilla == config.SEMILLA else f"{base}_s{semilla}"


def descripcion(hp):
    partes = "red completa" if hp["n_etapas"] >= 7 else f"últimas {hp['n_etapas']} etapas + cabeza"
    return (
        f"TensorFlow/Keras en GPU, orquestado con PySpark. Ajuste fino de EfficientNetB0 (ImageNet, Keras) "
        f"({partes}), vistas: {hp['vistas']}, entrada {hp['tamano'][0]}x{hp['tamano'][1]}; vaca = promedio de sus fotos."
    )


def construir_modelos(pred: DataFrame) -> DataFrame:
    base = F.concat(F.lit("tf_efficientnet_"), F.col("variante"))
    sueltos = pred.withColumn(
        "modelo",
        F.when(F.col("semilla") == config.SEMILLA, base).otherwise(F.concat(base, F.lit("_s"), F.col("semilla").cast("string"))),
    )
    n_sem = Window.partitionBy("variante")
    ensamble = (
        pred.withColumn("n_semillas", F.size(F.collect_set("semilla").over(n_sem)))
        .filter("n_semillas > 1")
        .groupBy("variante", "ruta", "id_vaca", "vista", "fold", "bcs")
        .agg(F.avg("pred").alias("pred"))
        .withColumn("modelo", F.concat(base, F.lit("_ensemble")))
    )
    posterior = (
        ensamble.filter("vista = 'posterior'")
        .withColumn("modelo", F.concat(F.col("modelo"), F.lit("_eval_posterior")))
    )
    cols = ["modelo", "variante", "ruta", "id_vaca", "vista", "fold", "bcs", "pred"]
    return sueltos.select(cols).unionByName(ensamble.select(cols)).unionByName(posterior.select(cols))


def evaluar(spark):
    pred = spark.read.parquet(str(config.PREDICCIONES / "cv"))
    evaluaciones = spark.read.parquet(str(config.EVALUACIONES_PARQUET))
    fotos = construir_modelos(pred).filter(~(F.col("modelo").endswith("_eval_posterior") & F.col("variante").endswith("_posterior"))).cache()
    pliegues = metricas_por_pliegue(fotos, evaluaciones).cache()
    res = resumen(pliegues).toPandas()
    infos = pred.select("variante", "semilla", "fold_prueba", "info").distinct().toPandas()
    variantes = {m: v for m, v in fotos.select("modelo", "variante").distinct().collect()}

    for _, r in res.iterrows():
        modelo, variante = r["modelo"], variantes[r["modelo"]]
        carpeta = config.RESULTADOS / "models" / modelo
        carpeta.mkdir(parents=True, exist_ok=True)
        f = fotos.filter(F.col("modelo") == modelo)
        pf = f.select("ruta", "id_vaca", "vista", "fold", "bcs", "pred").toPandas().sort_values(["id_vaca", "vista", "ruta"])
        pf.to_csv(carpeta / "photo_predictions.csv", index=False)
        pv = por_vaca(f).drop("modelo").toPandas().sort_values("id_vaca")[["id_vaca", "fold", "bcs", "pred", "n_fotos"]]
        pv.to_csv(carpeta / "cow_predictions.csv", index=False)
        pp = pliegues.filter(F.col("modelo") == modelo).drop("modelo").toPandas().sort_values("fold")
        pp.to_csv(carpeta / "fold_metrics.csv", index=False)

        hp = config.hiperparametros(variante)
        extra = {"hiperparametros": hp, "motor": "pyspark + tensorflow"}
        i = infos[infos["variante"] == variante]
        if not modelo.endswith(("_ensemble", "_eval_posterior")):
            semilla = config.SEMILLA if modelo == f"tf_efficientnet_{variante}" else int(modelo.rsplit("_s", 1)[1])
            extra["semilla"] = semilla
            extra["pliegues"] = {
                str(int(x["fold_prueba"])): {k: v for k, v in json.loads(x["info"]).items() if k != "historia"}
                for _, x in i[i["semilla"] == semilla].iterrows()
            }
            desc = descripcion(hp)
        else:
            semillas = sorted(i["semilla"].unique().tolist())
            extra["semillas"] = semillas
            desc = f"Promedio de {len(semillas)} semillas de: {descripcion(hp)}"
            if modelo.endswith("_eval_posterior"):
                desc += " Evaluado solo con las fotos posteriores (mismas 59 vacas que los modelos de vista posterior)."
        datos = {k: (None if v != v else v) for k, v in r.drop("modelo").to_dict().items()}
        res_json = {"modelo": modelo, "descripcion": desc, **datos, **extra}
        (carpeta / "summary.json").write_text(json.dumps(res_json, indent=2, ensure_ascii=False, default=float))
        print(f"{modelo}: MAE vaca {r['vaca_mae_media']:.3f} ± {r['vaca_mae_std']:.3f}, MAE foto {r['foto_mae_media']:.3f} ± {r['foto_mae_std']:.3f}", flush=True)

    for variante in sorted(set(variantes.values())):
        filas = res[res["modelo"].isin([nombre_modelo(variante, s) for s in infos[infos["variante"] == variante]["semilla"].unique()])]
        filas.to_csv(config.RESULTADOS / f"seeds_tf_efficientnet_{variante}.csv", index=False)
    print(res.sort_values("vaca_mae_media")[["modelo", "vaca_mae_media", "vaca_mae_std", "foto_mae_media"]].to_string(index=False))


if __name__ == "__main__":
    spark = config.crear_spark("bcs-evaluar")
    try:
        evaluar(spark)
    finally:
        spark.stop()
