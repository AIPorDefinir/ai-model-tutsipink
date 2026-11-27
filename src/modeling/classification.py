"""Modelo 5: logística multiclase sobre la rejilla de 0.25 con objetivo suave, elegida por CV anidada del TRAIN."""
import numpy as np
import pandas as pd

from . import config
from .metrics import dentro, pesos_vaca

CS = [float(10 ** e) for e in np.arange(-4, 1.01, 0.5)]
PESOS_CLASE = ["ninguno", "balanceado"]
DECODIFICACIONES = ("valor_esperado", "clase_mas_probable")


def clases_de(y_train, paso=config.PASO_BCS):
    lo = np.floor(np.min(y_train) / paso + config.EPS) * paso
    hi = np.ceil(np.max(y_train) / paso - config.EPS) * paso
    return np.round(np.arange(lo, hi + paso / 2, paso), 4)


def objetivos_suaves(y, clases):
    T = np.zeros((len(y), len(clases)))
    for i, v in enumerate(np.asarray(y, float)):
        d = np.abs(clases - v)
        if d.min() < config.EPS:
            T[i, d.argmin()] = 1.0
        elif v < clases[0] or v > clases[-1]:
            T[i, d.argmin()] = 1.0
        else:
            lo = int(np.flatnonzero(clases < v).max())
            t = (v - clases[lo]) / (clases[lo + 1] - clases[lo])
            T[i, lo], T[i, lo + 1] = 1.0 - t, t
    return T


def ajustar_predecir(X, y, vacas, Xe, C, peso_clase, clases):
    """Probabilidades (len(Xe), len(clases)); clases sin masa en el ajuste quedan con probabilidad 0."""
    from scipy.optimize import minimize

    T = objetivos_suaves(y, clases)
    con_masa = T.sum(0) > 0
    T = T[:, con_masa]
    w = pesos_vaca(vacas)
    if peso_clase == "balanceado":
        masa_vaca = pd.DataFrame(T).groupby(np.asarray(vacas)).mean()
        masa = masa_vaca.sum(0).to_numpy()
        w = w * (T @ (len(masa_vaca) / (T.shape[1] * masa)))
    w = w / w.mean()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z, Ze = (X - mu) / sd, (Xe - mu) / sd
    d, c = Z.shape[1], T.shape[1]

    def f(theta):
        W, b = theta[: d * c].reshape(d, c), theta[d * c:]
        L = Z @ W + b
        L = L - L.max(1, keepdims=True)
        logp = L - np.log(np.exp(L).sum(1, keepdims=True))
        perdida = C * float((w * -(T * logp).sum(1)).sum()) + 0.5 * float((W * W).sum())
        G = C * w[:, None] * (np.exp(logp) * T.sum(1, keepdims=True) - T)
        return perdida, np.concatenate([(Z.T @ G + W).ravel(), G.sum(0)])

    res = minimize(f, np.zeros(d * c + c), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "gtol": 1e-9, "ftol": 1e-12})
    W, b = res.x[: d * c].reshape(d, c), res.x[d * c:]
    L = Ze @ W + b
    P = np.exp(L - L.max(1, keepdims=True))
    P = P / P.sum(1, keepdims=True)
    salida = np.zeros((len(Xe), len(clases)))
    salida[:, con_masa] = P
    return salida


def decodificar_vacas(P, vacas, y, clases):
    df = pd.DataFrame(P, columns=range(len(clases))).assign(c=np.asarray(vacas), y=np.asarray(y, float))
    g = df.groupby("c", sort=True)
    Pm = g[list(range(len(clases)))].mean().to_numpy()
    return g["y"].first().index.to_numpy(), g["y"].first().to_numpy(), clases[Pm.argmax(1)], Pm @ clases


def puntaje(p, y):
    e = np.asarray(p, float) - np.asarray(y, float)
    return dentro(e), float(np.abs(e).mean())


def vacas_de(P, vacas, y, clases, dec):
    c, yc, am, vesp = decodificar_vacas(P, vacas, y, clases)
    return pd.DataFrame({"id_vaca": c, "bcs": yc, "pred": am if dec == "clase_mas_probable" else vesp})


def seleccion_anidada(E, y, vacas, fold, tr, ev, pr=None, cs=CS, pesos_clase=PESOS_CLASE):
    """Devuelve {decodificación: DataFrame(id_vaca, bcs, pred)} para las vacas de `ev`, la elección interna, y lo mismo
    para las vacas de `pr`."""
    clases = clases_de(y[tr])
    internos = sorted(set(fold[tr]))
    tabla = {d: {} for d in DECODIFICACIONES}
    for C in cs:
        for pc in pesos_clase:
            pa, pe, yy = [], [], []
            for j in internos:
                a, b = tr & (fold != j), tr & (fold == j)
                P = ajustar_predecir(E[a], y[a], vacas[a], E[b], C, pc, clases)
                _, yc, am, vesp = decodificar_vacas(P, vacas[b], y[b], clases)
                pa.append(am); pe.append(vesp); yy.append(yc)
            yy = np.concatenate(yy)
            tabla["clase_mas_probable"][(C, pc)] = puntaje(np.concatenate(pa), yy)
            tabla["valor_esperado"][(C, pc)] = puntaje(np.concatenate(pe), yy)
    pr = np.zeros(len(y), bool) if pr is None else pr
    salida, eleccion, prueba = {}, {}, {}
    for dec in DECODIFICACIONES:
        t = tabla[dec]
        mejor = min(t, key=lambda k: (-t[k][0], round(t[k][1], 9), k[0], pesos_clase.index(k[1])))
        P = ajustar_predecir(E[tr], y[tr], vacas[tr], np.vstack([E[ev], E[pr]]), *mejor, clases)
        n = int(ev.sum())
        salida[dec] = vacas_de(P[:n], vacas[ev], y[ev], clases, dec)
        if pr.any():
            prueba[dec] = vacas_de(P[n:], vacas[pr], y[pr], clases, dec)
        eleccion[dec] = {"C": mejor[0], "peso_clase": mejor[1], "vacas_025_interno": t[mejor][0],
                         "mae_interno": t[mejor][1]}
    return salida, {"clases": clases.tolist(), "eleccion": eleccion}, prueba
