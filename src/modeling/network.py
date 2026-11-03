import io
import json
import time

import numpy as np

from . import config


def configurar_gpu(precision="float32"):
    import keras
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    keras.mixed_precision.set_global_policy(precision)
    return [g.name for g in tf.config.list_physical_devices("GPU")]


def decodificar(jpgs, lado=config.LADO_FOTO):
    from PIL import Image

    alto, ancho = lado, lado * 3 // 4
    salida = np.empty((len(jpgs), alto, ancho, 3), dtype=np.uint8)
    for i, b in enumerate(jpgs):
        with Image.open(io.BytesIO(b)) as im:
            im = im.convert("RGB")
            if im.size != (ancho, alto):
                im = im.resize((ancho, alto), Image.Resampling.LANCZOS)
            salida[i] = np.asarray(im)
    return salida


def crear_modelo(hp, centro, pesos="imagenet"):
    import keras

    alto, ancho = hp["tamano"]
    base = keras.applications.EfficientNetB0(
        include_top=False, weights=pesos, input_shape=(alto, ancho, 3), pooling="avg"
    )
    etapas = tuple(f"block{i}" for i in range(8 - hp["n_etapas"], 8))
    for capa in base.layers:
        entrenable = capa.name.startswith(etapas) or capa.name.startswith("top_")
        if hp["n_etapas"] >= 7:
            entrenable = entrenable or capa.name.startswith("stem_")
        if isinstance(capa, keras.layers.BatchNormalization):
            entrenable = False
        capa.trainable = entrenable
    x = keras.layers.Dropout(hp["dropout"], name="dropout_cabeza")(base.output)
    x = keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros", dtype="float32", name="cabeza"
    )(x)
    salida = keras.layers.Rescaling(1.0, offset=float(centro), dtype="float32", name="centro")(x)
    return keras.Model(base.input, salida, name=f"bcs_{hp['red']}_e{hp['n_etapas']}")


def _aumentar(tamano):
    import keras
    import tensorflow as tf

    rotar = keras.layers.RandomRotation(8 / 360, fill_mode="constant", fill_value=0.0)

    def f(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        inicio, tam, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(img),
            bounding_boxes=tf.constant([[[0.0, 0.0, 1.0, 1.0]]]),
            min_object_covered=0.0,
            aspect_ratio_range=(0.68, 0.82),
            area_range=(0.7, 1.0),
            max_attempts=20,
            use_image_if_no_bounding_boxes=True,
        )
        img = tf.image.resize(tf.slice(img, inicio, tam), tamano, antialias=True)
        img = tf.image.random_flip_left_right(img)
        img = rotar(img[None], training=True)[0]
        img = img * tf.random.uniform([], 0.75, 1.25)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.02)
        return tf.clip_by_value(img, 0.0, 1.0) * 255.0

    return f


def _limpia(tamano):
    import tensorflow as tf

    def f(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, tamano, antialias=True) * 255.0

    return f


def pesos_por_foto(id_vaca, tipo="vaca"):
    id_vaca = np.asarray(id_vaca)
    if tipo == "foto":
        w = np.ones(len(id_vaca))
    else:
        _, inv, cuenta = np.unique(id_vaca, return_inverse=True, return_counts=True)
        w = 1.0 / cuenta[inv]
    return (w / w.mean()).astype(np.float32)


def predecir(modelo, X, hp, lote=32):
    import tensorflow as tf

    limpia = _limpia(tuple(hp["tamano"]))
    preds = []
    for i in range(0, len(X), lote):
        x = tf.map_fn(limpia, tf.constant(X[i : i + lote]), fn_output_signature=tf.float32)
        p = (modelo(x, training=False) + modelo(tf.image.flip_left_right(x), training=False)) / 2
        preds.append(np.asarray(p, dtype=np.float64)[:, 0])
    return np.clip(np.concatenate(preds), config.BCS_MIN, config.BCS_MAX)


def mae_vaca(id_vaca, y, pred):
    import pandas as pd

    v = pd.DataFrame({"id_vaca": id_vaca, "y": y, "pred": pred}).groupby("id_vaca")[["y", "pred"]].mean()
    return float((v["pred"] - v["y"]).abs().mean())


def entrenar(hp, X, y, id_vaca, epocas, val=None, semilla=config.SEMILLA, registro=print):
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(semilla)
    gen = np.random.default_rng(semilla)
    y = np.asarray(y, dtype=np.float32)
    w = pesos_por_foto(id_vaca, hp["peso"])
    modelo = crear_modelo(hp, centro=float(y.mean()))
    cabeza = modelo.get_layer("cabeza").trainable_variables
    ids_cabeza = {id(v) for v in cabeza}
    cola = [v for v in modelo.trainable_variables if id(v) not in ids_cabeza]
    variables = cola + cabeza
    pasos_epoca = int(np.ceil(len(y) / hp["lote"]))

    opt_cola = keras.optimizers.AdamW(hp["lr_cola"], weight_decay=hp["weight_decay"], epsilon=1e-8)
    opt_cabeza = keras.optimizers.AdamW(hp["lr_cabeza"], weight_decay=hp["weight_decay"], epsilon=1e-8)
    paso = 0

    beta = hp["huber_beta"]

    @tf.function(reduce_retracing=True)
    def gradientes(x, yb, wb):
        with tf.GradientTape() as tape:
            pred = tf.cast(modelo(x, training=True)[:, 0], tf.float32)
            err = tf.abs(pred - yb)
            huber = tf.where(err < beta, 0.5 * err**2 / beta, err - 0.5 * beta)
            perdida = tf.reduce_sum(huber * wb)
        return perdida, tape.gradient(perdida, variables)

    aumentar = _aumentar(tuple(hp["tamano"]))
    Xtf = tf.constant(X)
    ytf, wtf = tf.constant(y), tf.constant(w)

    def lotes(orden):
        ds = tf.data.Dataset.from_tensor_slices(orden.astype(np.int32))
        ds = ds.map(lambda i: (aumentar(tf.gather(Xtf, i)), tf.gather(ytf, i), tf.gather(wtf, i)), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(hp["microlote"]).prefetch(2)

    por_lote = hp["lote"] // hp["microlote"]
    historia, mejor = [], {"mae": np.inf, "epoca": 0, "pesos": None}
    for epoca in range(1, epocas + 1):
        t0 = time.time()
        orden = gen.permutation(len(y))
        perdida_total, acum, n_lote, cuenta = 0.0, None, 0, 0
        n_micro = int(np.ceil(len(y) / hp["microlote"]))
        for j, (x, yb, wb) in enumerate(lotes(orden)):
            perdida, grads = gradientes(x, yb, wb)
            acum = grads if acum is None else [a + g for a, g in zip(acum, grads)]
            n_lote += int(x.shape[0])
            perdida_total += float(perdida)
            cuenta += 1
            if cuenta == por_lote or j == n_micro - 1:
                factor = min(1.0, (paso + 1) / pasos_epoca)
                opt_cola.learning_rate = hp["lr_cola"] * factor
                opt_cabeza.learning_rate = hp["lr_cabeza"] * factor
                grads = [g / n_lote for g in acum]
                opt_cola.apply_gradients(zip(grads[: len(cola)], cola))
                opt_cabeza.apply_gradients(zip(grads[len(cola) :], cabeza))
                paso += 1
                acum, n_lote, cuenta = None, 0, 0
        fila = {"epoca": epoca, "perdida": perdida_total / len(y)}
        if val is not None:
            Xv, yv, iv = val
            fila["mae_vaca_val"] = mae_vaca(iv, yv, predecir(modelo, Xv, hp))
            if fila["mae_vaca_val"] < mejor["mae"] - 1e-4:
                mejor = {"mae": fila["mae_vaca_val"], "epoca": epoca, "pesos": [np.copy(p) for p in modelo.get_weights()]}
        fila["segundos"] = round(time.time() - t0, 1)
        historia.append(fila)
        registro(json.dumps(fila))
        if val is not None and epoca - mejor["epoca"] >= hp["paciencia"]:
            break
    if val is not None:
        modelo.set_weights(mejor["pesos"])
    else:
        mejor["epoca"] = epocas
    return modelo, mejor["epoca"], historia
