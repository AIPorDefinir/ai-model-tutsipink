# Modelo 5 empaquetado en models/modelo5_bcs.keras (solo capas estándar de Keras):
#   python -m src.modeling.modelo5 foto1.jpg foto2.jpg foto3.jpg
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

MODELO = config.RAIZ / "models" / "modelo5_bcs.keras"
CLASES = np.arange(4, 21) / 4.0


def cargar(ruta=MODELO):
    import keras

    return keras.models.load_model(ruta)


def predecir(modelo, X, id_vaca, lote=4):
    # X: fotos segmentadas en RGB de 512 x 384; el volteo horizontal ya va dentro del .keras
    P = modelo.predict(np.asarray(X, np.float32), batch_size=lote, verbose=0)
    g = pd.DataFrame(P).groupby(np.asarray(id_vaca).astype(str), sort=True).mean()
    esperado = g.to_numpy() @ CLASES
    return pd.DataFrame({"id_vaca": g.index, "valor_esperado": esperado, "bcs": np.floor(esperado * 4 + 0.5) / 4})


if __name__ == "__main__":
    from .network import decodificar

    p = argparse.ArgumentParser(description="BCS de una vaca a partir de sus fotos segmentadas")
    p.add_argument("fotos", nargs="+")
    a = p.parse_args()
    X = decodificar([Path(f).read_bytes() for f in a.fotos])
    v = predecir(cargar(), X, ["vaca"] * len(X)).iloc[0]
    print(f"BCS {v['bcs']:.2f} (valor esperado {v['valor_esperado']:.3f})")
