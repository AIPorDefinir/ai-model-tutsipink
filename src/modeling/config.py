"""Rutas, constantes y configuraciones del modelado. Las rutas a datos salen de variables de entorno."""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CONFIGS = Path(__file__).resolve().parent / "configs"

DATOS = Path(os.environ.get("BCS_DATOS", RAIZ / "data"))
DATASET_LOCAL = Path(os.environ.get("BCS_DATASET_LOCAL", DATOS / "dataset_bcs"))
DATOS_SPARK = DATOS / "spark"
FOTOS_PARQUET = DATOS_SPARK / "fotos"
VACAS_PARQUET = DATOS_SPARK / "vacas"
EMBEDDINGS = DATOS_SPARK / "embeddings"
PREDICCIONES = DATOS_SPARK / "predicciones"
RESULTADOS = Path(os.environ.get("BCS_RESULTADOS", RAIZ / "results"))
ARTEFACTOS = Path(os.environ.get("BCS_ARTEFACTOS", RAIZ / "artifacts"))
REGISTROS = ARTEFACTOS / "registros"
AJUSTE_FINO = ARTEFACTOS / "ajuste_fino"

SEGMENTADOR = Path(os.environ.get(
    "BCS_SEGMENTADOR",
    ARTEFACTOS / "segmentador" / "mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8" / "saved_model"))
PESOS_RESNET50_VACAS = Path(os.environ.get(
    "BCS_PESOS_RESNET50_VACAS", ARTEFACTOS / "pesos" / "class_12_box_large.pth"))

PREFIJO_DATASET = os.environ.get("BCS_PREFIJO", "bcs").strip("/")
REGION = os.environ.get("AWS_DEFAULT_REGION", "mx-central-1")
HADOOP_AWS = os.environ.get("BCS_HADOOP_AWS", "org.apache.hadoop:hadoop-aws:3.5.0")

LADO_FOTO = 512
VISTAS = ["posterior", "lateral_izq", "lateral_der"]
K = 5
SEMILLA = 2026
BCS_MIN, BCS_MAX = 1.0, 5.0
PASO_BCS = 0.25
EPS = 1e-9
ENTRADAS = ("original", "segmentada")
PROTOCOLOS = ("laboratorio", "cv5")

COLUMNAS = {
    "ruta": "ruta",
    "id_vaca": "id_vaca",
    "vista": "vista",
    "fold": "fold",
    "bcs": "bcs_entrenamiento",
    "usar_foto": "usar_en_entrenamiento",
    "usar_vaca": "en_entrenamiento",
}

MODELOS = {
    "cnn_propia": "Modelo 1: CNN propia",
    "convnext_tiny": "Modelo 2: ConvNeXt-Tiny preentrenada (congelada + regresión)",
    "eva02_ajuste_fino": "Modelo 3: EVA02 ajuste fino",
    "resnet50_vacas": "Modelo 4: ResNet-50 preentrenada en vacas (externa)",
    "clasificacion_bcs": "Modelo 5: clasificación por clases BCS",
}

HIPERPARAMETROS_CNN = {
    "tamano": [384, 288],
    "epocas_max": 40,
    "paciencia": 10,
    "lote": 32,
    "microlote": 16,
    "lr_cuerpo": 3e-4,
    "lr_cabeza": 1e-3,
    "weight_decay": 0.05,
    "huber_beta": 0.25,
    "dropout": 0.3,
    "dtype": "float32",
}


def hiperparametros_cnn(**cambios):
    return {**HIPERPARAMETROS_CNN, **cambios}


def cargar_config(ruta):
    """Lee la configuración YAML de un modelo (ruta o nombre corto) y completa los valores por omisión."""
    import yaml

    ruta = Path(ruta)
    if not ruta.is_file():
        nombre = ruta.name if ruta.suffix else f"{ruta.name}.yaml"
        hallados = [CONFIGS / nombre] if (CONFIGS / nombre).is_file() else sorted(CONFIGS.glob(f"*_{nombre}"))
        if len(hallados) != 1:
            raise FileNotFoundError(f"No se encontró la configuración {ruta} (ni en {CONFIGS})")
        ruta = hallados[0]
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    cfg.setdefault("protocolo", "laboratorio")
    cfg.setdefault("semilla", SEMILLA)
    cfg.setdefault("agregacion_vaca", "media")
    cfg.setdefault("entrada", "segmentada")
    if cfg["entrada"] not in ENTRADAS:
        raise ValueError(f"entrada desconocida: {cfg['entrada']}")
    if cfg["protocolo"] not in PROTOCOLOS:
        raise ValueError(f"protocolo desconocido: {cfg['protocolo']}")
    oficial = cfg.get("lectura_oficial")
    if oficial and oficial["salida"] not in cfg.get("salidas", ["continua"]):
        raise ValueError(f"la lectura oficial {oficial['salida']} no está entre las salidas del modelo")
    return cfg


def fotos_parquet(entrada="original"):
    return FOTOS_PARQUET if entrada == "original" else DATOS_SPARK / f"fotos_{entrada}"


def cargar_env():
    """Pasa al proceso las variables de un .env local si existe, sin imprimirlas."""
    os.environ.pop("AWS_PROFILE", None)
    archivo = RAIZ / ".env"
    if archivo.is_file():
        from dotenv import load_dotenv

        load_dotenv(archivo, override=True)


def bucket():
    cargar_env()
    return os.environ["S3_BUCKET"]


def crear_spark(nombre, s3=False, una_tarea_a_la_vez=False, memoria="6g", nucleos=None):
    """Sesión local de Spark; con una_tarea_a_la_vez cada tarea ocupa todos los núcleos y no se comparte la GPU."""
    from pyspark.sql import SparkSession

    nucleos = nucleos or os.cpu_count() or 4
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
