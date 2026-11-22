"""Punto de entrada del modelado: corre las etapas de un modelo descrito por su configuración YAML.

  python -m src.modeling.pipeline --config src/modeling/configs/modelo4_resnet50_vacas.yaml
  python -m src.modeling.pipeline --config resnet50_vacas --etapas entrenar evaluar
"""
import argparse
import json
import os
import time

from . import config

ETAPAS = ("preparar", "segmentar", "embeddings", "entrenar", "evaluar")
CON_GPU = {"cnn", "ajuste_fino"}


def etapas_de(cfg):
    etapas = ["preparar"]
    if cfg["entrada"] == "segmentada":
        etapas.append("segmentar")
    if cfg["tipo"].startswith("embeddings_"):
        etapas.append("embeddings")
    return etapas + ["entrenar", "evaluar"]


def tareas_de(cfg):
    cfg_json = json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")})
    if cfg["tipo"] == "ajuste_fino":
        from .finetune import SEMILLAS

        return [{"cfg": cfg_json, "k": k, "semilla": s} for s in cfg.get("semillas", SEMILLAS) for k in range(config.K)]
    return [{"cfg": cfg_json, "k": k, "semilla": int(cfg["semilla"])} for k in range(config.K)]


def correr(cfg, etapas, s3=False, local=None, forzar=False):
    from . import evaluate, prepare, tasks

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    gpu = cfg["tipo"] in CON_GPU or "segmentar" in etapas or "embeddings" in etapas
    spark = config.crear_spark(f"bcs-{cfg['nombre']}", s3=s3, una_tarea_a_la_vez=gpu)
    t0 = time.time()
    try:
        if "preparar" in etapas:
            if config.FOTOS_PARQUET.exists() and not forzar:
                print(f"Fotos preparadas ya existen en {config.FOTOS_PARQUET}; usa --forzar para rehacerlas", flush=True)
            else:
                prepare.preparar(spark, prepare.base_datos(s3, local))
        if "segmentar" in etapas and cfg["entrada"] == "segmentada":
            from . import segmentation

            segmentation.segmentar(spark, forzar)
        if "embeddings" in etapas and cfg["tipo"].startswith("embeddings_"):
            from . import embeddings

            embeddings.extraer(spark, cfg["extractor"], cfg["entrada"], forzar)
        destino = config.PREDICCIONES / cfg["nombre"]
        if "entrenar" in etapas:
            tareas = tareas_de(cfg)
            print(f"{config.MODELOS.get(cfg['nombre'], cfg['nombre'])}: {len(tareas)} tareas en Spark", flush=True)
            tasks.lanzar(spark, tareas, destino)
            if cfg["tipo"] == "ajuste_fino":
                from .finetune import combinar

                combinar(spark, cfg, destino)
        if "evaluar" in etapas:
            return evaluate.evaluar(spark, cfg["nombre"], cfg.get("salidas"), cfg.get("lectura_oficial"))
    finally:
        spark.stop()
        print(f"Listo en {time.time() - t0:.0f} s", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Modelado BCS de punta a punta, orquestado con PySpark")
    p.add_argument("--config", required=True, help="YAML del modelo (ruta o nombre dentro de src/modeling/configs/)")
    p.add_argument("--etapas", nargs="+", choices=ETAPAS, help="por omisión, todas las que aplican al modelo")
    p.add_argument("--s3", action="store_true", help="ingesta directa del bucket (s3a) en lugar de la copia local")
    p.add_argument("--local", help="carpeta local del dataset (por omisión $BCS_DATASET_LOCAL)")
    p.add_argument("--protocolo", choices=config.PROTOCOLOS, help="sobrescribe el protocolo de la configuración")
    p.add_argument("--forzar", action="store_true", help="rehacer etapas aunque su salida exista")
    a = p.parse_args(argv)
    cfg = config.cargar_config(a.config)
    if a.protocolo:
        cfg["protocolo"] = a.protocolo
    from pathlib import Path

    correr(cfg, a.etapas or etapas_de(cfg), a.s3, Path(a.local) if a.local else None, a.forzar)


if __name__ == "__main__":
    main()
