"""Papel de cada pliegue por vaca en el pliegue externo k, según el protocolo «laboratorio» o «cv5»."""
import numpy as np

from . import config


def pliegue_evaluado(k, protocolo="laboratorio", n=config.K):
    if protocolo == "laboratorio":
        return (k + 1) % n
    if protocolo == "cv5":
        return k
    raise ValueError(f"protocolo desconocido: {protocolo}")


def pliegues_train(k, protocolo="laboratorio", n=config.K):
    ev = pliegue_evaluado(k, protocolo, n)
    fuera = {k, ev} if protocolo == "laboratorio" else {ev}
    return [j for j in range(n) if j not in fuera]


def particion(fold, k, protocolo="laboratorio", n=config.K):
    """Máscaras booleanas (evaluado, train) por foto para el pliegue externo k."""
    fold = np.asarray(fold).astype(int)
    ev = fold == pliegue_evaluado(k, protocolo, n)
    tr = np.isin(fold, pliegues_train(k, protocolo, n))
    return ev, tr


def prueba(fold, k, protocolo="laboratorio"):
    # En «laboratorio» el pliegue k no entra al TRAIN ni a ninguna elección; solo se predice al final
    fold = np.asarray(fold).astype(int)
    return fold == k if protocolo == "laboratorio" else np.zeros(len(fold), bool)


def pliegue_paro(k, protocolo="laboratorio", n=config.K):
    """Pliegue interno del TRAIN que decide la época de paro: el siguiente al evaluado."""
    return (pliegue_evaluado(k, protocolo, n) + 1) % n


def revisar(id_vaca, ev, tr):
    """Falla si una vaca o una foto queda a la vez en TRAIN y en el pliegue evaluado."""
    id_vaca = np.asarray(id_vaca)
    if set(id_vaca[ev]) & set(id_vaca[tr]):
        raise ValueError("una vaca aparece en TRAIN y en el pliegue evaluado")
    if (ev & tr).any():
        raise ValueError("una foto aparece en TRAIN y en el pliegue evaluado")
