"""Segmentación de la vaca con Mask R-CNN de COCO (clase 21) por lotes con Spark; el fondo queda en gris neutro."""
import io
import math

import numpy as np

from . import config

CLASE_VACA = 21
UMBRAL = 0.5
UMBRAL_BAJO = 0.3
UMBRAL_MASCARA = 0.5
FRACCION_MIN_CAJA = 0.05
FRACCION_MIN_MASCARA = 0.02
DILATACION = 11
GRIS = (124, 116, 104)
CALIDAD_JPEG = 95
EXITO = ("detectada", "detectada_baja")
ESQUEMA = "ruta string, id_vaca string, vista string, fold int, bcs double, jpg binary, estado string"


def cargar_segmentador(ruta=config.SEGMENTADOR):
    import tensorflow as tf

    if not (ruta / "saved_model.pb").is_file():
        raise FileNotFoundError(f"No se encontró el SavedModel del segmentador en {ruta} (variable BCS_SEGMENTADOR)")
    return tf.saved_model.load(str(ruta))


def decodificar_jpg(blob):
    from PIL import Image

    with Image.open(io.BytesIO(blob)) as im:
        return np.asarray(im.convert("RGB"))


def codificar_jpg(imagen):
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(imagen).save(out, "JPEG", quality=CALIDAD_JPEG)
    return out.getvalue()


def mascara_completa(mascara, caja, alto, ancho):
    if mascara.shape == (alto, ancho):
        return mascara >= UMBRAL_MASCARA
    from PIL import Image

    y0, x0, y1, x1 = caja
    arriba, izq = max(0, math.floor(y0)), max(0, math.floor(x0))
    abajo, der = min(alto, math.ceil(y1)), min(ancho, math.ceil(x1))
    llena = np.zeros((alto, ancho), dtype=bool)
    if abajo - arriba < 1 or der - izq < 1:
        return llena
    r = Image.fromarray(mascara.astype(np.float32), mode="F").resize((der - izq, abajo - arriba), Image.Resampling.BILINEAR)
    llena[arriba:abajo, izq:der] = np.asarray(r) >= UMBRAL_MASCARA
    return llena


def detectar(modelo, imagen):
    import tensorflow as tf

    out = modelo(tf.constant(imagen[None]))
    n = int(out["num_detections"][0])
    clases = out["detection_classes"][0][:n].numpy().astype(int)
    puntajes = out["detection_scores"][0][:n].numpy()
    cajas = out["detection_boxes"][0][:n].numpy()
    mascaras = out["detection_masks"][0][:n].numpy() if "detection_masks" in out else None
    alto, ancho = imagen.shape[:2]
    halladas = []
    for i in range(n):
        if clases[i] != CLASE_VACA or puntajes[i] < UMBRAL_BAJO:
            continue
        caja = tuple(float(v) for v in cajas[i] * [alto, ancho, alto, ancho])
        m = None if mascaras is None else mascara_completa(mascaras[i], caja, alto, ancho)
        halladas.append({"puntaje": float(puntajes[i]), "caja": caja, "mascara": m})
    return halladas


def area(caja):
    y0, x0, y1, x1 = caja
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)


def elegir_deteccion(halladas, alto, ancho):
    """La detección de mayor área entre las de puntaje >= UMBRAL (o, si no hay, entre las de puntaje bajo)."""
    fuertes = [d for d in halladas if d["puntaje"] >= UMBRAL]
    grupo, estado = (fuertes, "detectada") if fuertes else (halladas, "detectada_baja")
    if not grupo:
        return "sin_deteccion", None
    elegida = max(grupo, key=lambda d: (area(d["caja"]), d["puntaje"]))
    if area(elegida["caja"]) < FRACCION_MIN_CAJA * alto * ancho:
        return "deteccion_pequena", elegida
    return estado, elegida


def fondo_neutro(imagen, mascara):
    from PIL import Image, ImageFilter

    crecida = Image.fromarray(mascara.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(DILATACION))
    salida = imagen.copy()
    salida[~(np.asarray(crecida) > 0)] = GRIS
    return salida


def procesar(modelo, blob):
    """Devuelve (estado, jpg segmentado); si falla, (estado, jpg original)."""
    try:
        imagen = decodificar_jpg(blob)
    except Exception:
        return "error_decodificacion", blob
    alto, ancho = imagen.shape[:2]
    try:
        estado, elegida = elegir_deteccion(detectar(modelo, imagen), alto, ancho)
    except Exception:
        return "error_modelo", blob
    if estado not in EXITO:
        return estado, blob
    if elegida["mascara"] is None:
        return "sin_mascara", blob
    if float(elegida["mascara"].mean()) < FRACCION_MIN_MASCARA:
        return "mascara_vacia", blob
    return estado, codificar_jpg(fondo_neutro(imagen, elegida["mascara"]))


def _segmentar_lotes(lotes):
    from .network import configurar_gpu

    configurar_gpu("float32")
    modelo = cargar_segmentador()
    for lote in lotes:
        estados, jpgs = zip(*(procesar(modelo, b) for b in lote["jpg"])) if len(lote) else ((), ())
        yield lote.assign(jpg=list(jpgs), estado=list(estados))


def segmentar(spark, forzar=False):
    from pyspark.sql import functions as F

    destino = config.fotos_parquet("segmentada")
    if destino.exists() and not forzar:
        print(f"Fotos segmentadas ya existen en {destino}; usa --forzar para rehacerlas", flush=True)
        return destino
    fotos = spark.read.parquet(str(config.FOTOS_PARQUET)).select("ruta", "id_vaca", "vista", "fold", "bcs", "jpg")
    seg = fotos.repartition(config.K, "fold").mapInPandas(_segmentar_lotes, ESQUEMA)
    seg.write.mode("overwrite").partitionBy("fold").parquet(str(destino))
    guardadas = spark.read.parquet(str(destino))
    exito = F.col("estado").isin(*EXITO).cast("int")
    guardadas.groupBy("fold").agg(F.count("*").alias("fotos"), F.sum(exito).alias("segmentadas")).orderBy("fold").show()
    guardadas.groupBy("estado").count().orderBy("estado").show()
    return destino
