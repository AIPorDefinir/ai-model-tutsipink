"""Embeddings congelados por lotes con Spark (mapInPandas); cada foto se promedia con su reflejo horizontal.

Extractores: convnext_tiny (Modelo 2) y resnet50_vacas (Modelos 4 y 5).
"""
import numpy as np

from . import config

ESQUEMA = "ruta string, id_vaca string, vista string, fold int, bcs double, emb array<float>"
TAMANO_CONVNEXT = (384, 288)
LOTE = 16


def ruta_embeddings(extractor, entrada):
    return config.EMBEDDINGS / f"{extractor}_{entrada}"


def extractor_convnext_tiny():
    import keras
    import tensorflow as tf

    from .network import _limpia, configurar_gpu

    configurar_gpu("float32")
    alto, ancho = TAMANO_CONVNEXT
    base = keras.applications.ConvNeXtTiny(include_top=False, weights="imagenet", input_shape=(alto, ancho, 3),
                                           pooling="avg")
    limpia = _limpia(TAMANO_CONVNEXT)

    def f(X):
        x = tf.map_fn(limpia, tf.constant(X), fn_output_signature=tf.float32)
        return np.asarray((base(x, training=False) + base(tf.image.flip_left_right(x), training=False)) / 2, np.float32)
    return f


def resnet50_vacas(ruta=None):
    """Backbone ResNet-50 del detector de Nagy et al. (2023) con FrozenBN, sin Detectron2.
    Devuelve (red, media de píxel BGR del checkpoint)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FBN(nn.Module):
        def __init__(self, c):
            super().__init__()
            for n in ("weight", "bias", "running_mean", "running_var"):
                self.register_buffer(n, torch.zeros(c))

        def forward(self, x):
            s = self.weight * (self.running_var + 1e-5).rsqrt()
            return x * s.view(1, -1, 1, 1) + (self.bias - self.running_mean * s).view(1, -1, 1, 1)

    class Conv(nn.Conv2d):
        def __init__(self, cin, cout, k, stride=1, pad=0):
            super().__init__(cin, cout, k, stride, pad, bias=False)
            self.norm = FBN(cout)

        def forward(self, x):
            return self.norm(super().forward(x))

    class Bloque(nn.Module):
        def __init__(self, cin, cb, cout, stride):
            super().__init__()
            self.shortcut = Conv(cin, cout, 1, stride) if (cin != cout or stride != 1) else None
            self.conv1, self.conv2, self.conv3 = Conv(cin, cb, 1, stride), Conv(cb, cb, 3, 1, 1), Conv(cb, cout, 1)

        def forward(self, x):
            y = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
            return F.relu(y + (x if self.shortcut is None else self.shortcut(x)))

    class R50(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Module()
            self.stem.conv1 = Conv(3, 64, 7, 2, 3)
            etapas, cin = [(64, 256, 3, 1), (128, 512, 4, 2), (256, 1024, 6, 2), (512, 2048, 3, 2)], 64
            for i, (cb, cout, n, st) in enumerate(etapas):
                setattr(self, f"res{i + 2}", nn.Sequential(*[Bloque(cin if b == 0 else cout, cb, cout, st if b == 0 else 1)
                                                             for b in range(n)]))
                cin = cout
            self.num_features = 2048

        def forward_features(self, x):
            x = F.max_pool2d(F.relu(self.stem.conv1(x)), 3, 2, 1)
            return self.res5(self.res4(self.res3(self.res2(x))))

    ruta = ruta or config.PESOS_RESNET50_VACAS
    if not ruta.is_file():
        raise FileNotFoundError(f"Faltan los pesos de la ResNet-50 de vacas en {ruta} (variable BCS_PESOS_RESNET50_VACAS)")
    sd = torch.load(ruta, map_location="cpu", weights_only=False)["model"]
    red = R50()
    pref = "backbone.bottom_up."
    faltan, sobran = red.load_state_dict({k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}, strict=False)
    if faltan or sobran:
        raise ValueError(f"Pesos incompatibles: faltan {faltan}, sobran {sobran}")
    return red, sd["pixel_mean"].flatten().tolist()


def extractor_resnet50_vacas():
    import torch

    red, media = resnet50_vacas()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red = red.eval().to(dev)
    media = torch.tensor(media, device=dev).view(1, 3, 1, 1)

    def f(X):
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(X)).to(dev).permute(0, 3, 1, 2).float()
            x = x[:, [2, 1, 0]] - media

            def gap(z):
                return red.forward_features(z).mean((2, 3))

            return ((gap(x) + gap(torch.flip(x, dims=[3]))) / 2).float().cpu().numpy()
    return f


EXTRACTORES = {"convnext_tiny": extractor_convnext_tiny, "resnet50_vacas": extractor_resnet50_vacas}


def embeber(extractor_fn, jpgs, lote=LOTE):
    from .network import decodificar

    salida = [extractor_fn(decodificar(jpgs[i:i + lote])) for i in range(0, len(jpgs), lote)]
    E = np.concatenate(salida) if salida else np.zeros((0, 0), np.float32)
    if not np.isfinite(E).all():
        raise ValueError("embeddings con valores no finitos")
    return E


def _lotes_fn(nombre):
    def correr(lotes):
        extractor = EXTRACTORES[nombre]()
        for lote in lotes:
            if not len(lote):
                continue
            E = embeber(extractor, lote["jpg"].tolist())
            yield lote.drop(columns="jpg").assign(emb=[e.astype(np.float32) for e in E])
    return correr


def extraer(spark, extractor, entrada, forzar=False):
    destino = ruta_embeddings(extractor, entrada)
    if destino.exists() and not forzar:
        print(f"Embeddings ya existen en {destino}; usa --forzar para rehacerlos", flush=True)
        return destino
    fotos = spark.read.parquet(str(config.fotos_parquet(entrada))).select("ruta", "id_vaca", "vista", "fold", "bcs", "jpg")
    emb = fotos.repartition(config.K, "fold").mapInPandas(_lotes_fn(extractor), ESQUEMA)
    emb.write.mode("overwrite").partitionBy("fold").parquet(str(destino))
    guardados = spark.read.parquet(str(destino))
    n, dim = guardados.count(), len(guardados.first()["emb"])
    print(f"Embeddings {extractor}/{entrada}: {n} fotos x {dim} dimensiones en {destino}", flush=True)
    return destino


def cargar(extractor, entrada):
    """Lee los embeddings en el proceso actual (dentro de una tarea de Spark o en el driver)."""
    import pyarrow.parquet as pq

    t = pq.read_table(str(ruta_embeddings(extractor, entrada)), partitioning="hive").to_pandas()
    t["fold"] = t["fold"].astype(int)
    t["id_vaca"] = t["id_vaca"].astype(str)
    t = t.sort_values("ruta").reset_index(drop=True)
    E = np.stack(t["emb"].to_numpy()).astype(np.float64)
    return E, t.drop(columns="emb")
