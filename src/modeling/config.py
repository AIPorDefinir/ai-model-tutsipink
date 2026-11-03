import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATA = RAIZ / "data"
SPARK_DATA = DATA / "spark"
FOTOS_PARQUET = SPARK_DATA / "photos"
EVALUACIONES_PARQUET = SPARK_DATA / "evaluations"
VACAS_PARQUET = SPARK_DATA / "cows"
PREDICCIONES = SPARK_DATA / "predictions"
RESULTADOS = RAIZ / "results"
ARTEFACTOS = RAIZ / "artifacts" / "tensorflow"

PREFIJO_DATASET = os.environ.get("BCS_PREFIJO", "dataset_bcs").strip("/")
REGION = os.environ.get("AWS_DEFAULT_REGION", "mx-central-1")
HADOOP_AWS = os.environ.get("BCS_HADOOP_AWS", "org.apache.hadoop:hadoop-aws:3.5.0")

LADO_FOTO = 512
VISTAS = ["posterior", "lateral_izq", "lateral_der"]
K = 5
SEMILLA = 2026
SEMILLAS = [2026, 2027, 2028]
BCS_MIN, BCS_MAX = 1.0, 5.0

BASE = {
    "red": "efficientnet_b0",
    "vistas": "all",
    "n_etapas": 2,
    "tamano": [384, 288],
    "epocas_max": 40,
    "paciencia": 10,
    "lote": 32,
    "microlote": 16,
    "lr_cola": 3e-4,
    "lr_cabeza": 1e-3,
    "weight_decay": 0.05,
    "huber_beta": 0.25,
    "dropout": 0.3,
    "peso": "vaca",
}

VARIANTES = {
    "b0_last2_all": {},
    "b0_full_all": {"n_etapas": 7, "lr_cola": 1e-4},
    "b0_last2_posterior": {"vistas": "posterior"},
}


def hiperparametros(variante, **cambios):
    return {**BASE, **VARIANTES[variante], **cambios, "variante": variante}


def cargar_env():
    os.environ.pop("AWS_PROFILE", None)
    archivo = RAIZ / ".env"
    if archivo.is_file():
        from dotenv import load_dotenv

        load_dotenv(archivo, override=True)


def bucket():
    cargar_env()
    return os.environ["S3_BUCKET"]


def crear_spark(nombre, s3=False, una_tarea_a_la_vez=False, memoria="6g"):
    from pyspark.sql import SparkSession

    nucleos = os.cpu_count() or 4
    b = (
        SparkSession.builder.appName(nombre)
        .master(f"local[{nucleos}]")
        .config("spark.driver.memory", memoria)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    if s3:
        cargar_env()
        b = (
            b.config("spark.jars.packages", HADOOP_AWS)
            .config("spark.hadoop.fs.s3a.endpoint.region", REGION)
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "software.amazon.awssdk.auth.credentials.EnvironmentVariableCredentialsProvider",
            )
        )
    if una_tarea_a_la_vez:
        b = (
            b.config("spark.task.cpus", str(nucleos))
            .config("spark.python.worker.reuse", "false")
            .config("spark.task.maxFailures", "1")
            .config("spark.network.timeout", "3600s")
            .config("spark.executor.heartbeatInterval", "60s")
        )
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def ruta_s3a(*partes):
    return "s3a://" + "/".join([bucket(), PREFIJO_DATASET, *partes]).rstrip("/")
