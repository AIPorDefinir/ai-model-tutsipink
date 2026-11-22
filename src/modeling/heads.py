"""Regresión sobre embeddings congelados (Modelos 2 y 4): 25 recetas base × estiramiento, elegidas por CV anidada
dentro del TRAIN."""
import numpy as np
import pandas as pd

from . import config
from .metrics import dentro, errores_vaca, mae_vaca, pesos_vaca, por_vaca

ALFAS = np.logspace(-1, 5, 25).tolist()
BASES = ("ridge_min", "ridge_1se", "ridge_ens1se", "ridge_enstop5", "mlp")
ESTIRAMIENTOS = ("ninguno", "afin", "inverso", "isotonico", "pendiente")
PENDIENTES = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
SEMILLAS_MLP = [config.SEMILLA, config.SEMILLA + 1, config.SEMILLA + 2]
MLP_HP = {"ocultas": 256, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-4, "lote": 64, "epocas_max": 300,
          "paciencia": 30, "huber_beta": 0.25}
MLP_REJILLA = [dict(ocultas=h, dropout=d, weight_decay=wd, perdida=pe)
               for h in (64, 256) for d in (0.2, 0.5) for wd in (1e-4, 1e-1) for pe in ("huber", "mae")]


def aplicar_a_vaca(fn, id_vaca, p, agregacion="media"):
    """Transforma la predicción de la vaca y la reparte a sus fotos (la métrica es por vaca)."""
    g = "median" if agregacion == "mediana" else "mean"
    s = pd.Series(p).groupby(pd.Series(np.asarray(id_vaca))).transform(g).to_numpy()
    return np.clip(fn(s), config.BCS_MIN, config.BCS_MAX)


def ajustar_svd(X, y, w):
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w)) + 1e-6
    y0 = float(np.average(y, weights=w))
    raiz = np.sqrt(w)[:, None]
    U, s, Vt = np.linalg.svd(raiz * (X - mu) / sd, full_matrices=False)
    b = U.T @ (raiz[:, 0] * (y - y0))
    rango = int((s > s[0] * 1e-10).sum()) if s.size else 0
    return {"mu": mu, "sd": sd, "y0": y0, "s": s[:rango], "Vt": Vt[:rango], "b": b[:rango]}


def coef_ridge(fit, alfa):
    return fit["Vt"].T @ (fit["s"] / (fit["s"] ** 2 + alfa) * fit["b"])


def predecir_lineal(fit, beta, X):
    return np.clip((X - fit["mu"]) / fit["sd"] @ beta + fit["y0"], config.BCS_MIN, config.BCS_MAX)


def ridge_familia(X, y, w, Xe, alfas):
    fit = ajustar_svd(X, y, w)
    return {a: predecir_lineal(fit, coef_ridge(fit, a), Xe) for a in alfas}


def elegir_alfas(rejilla, curvas, regla):
    """curvas: {alfa: [MAE por pliegue interno]}; rejilla de menor a mayor regularización. Devuelve {alfa: peso}."""
    media = {h: float(np.mean(v)) for h, v in curvas.items()}
    hmin = min(rejilla, key=lambda h: media[h])
    n = len(curvas[hmin])
    se = float(np.std(curvas[hmin], ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if regla == "min":
        return {hmin: 1.0}, media, se
    ok = [h for h in rejilla if media[h] <= media[hmin] + se + config.EPS]
    if regla == "1se":
        return {max(ok): 1.0}, media, se
    if regla == "ens1se":
        return {h: 1.0 / len(ok) for h in ok}, media, se
    if regla == "enstop5":
        top = sorted(rejilla, key=lambda h: media[h])[:5]
        return {h: 1.0 / len(top) for h in top}, media, se
    raise ValueError(regla)


def modelo_ridge(regla, alfas=ALFAS):
    def correr(E, y, vacas, fold, tr, ev, agregacion="media"):
        internos = sorted(set(fold[tr]))
        curvas = {h: [] for h in alfas}
        oof_h = {h: np.full(tr.sum(), np.nan) for h in alfas}
        pos = np.flatnonzero(tr)
        for j in internos:
            itr, iva = tr & (fold != j), tr & (fold == j)
            preds = ridge_familia(E[itr], y[itr], pesos_vaca(vacas[itr]), E[iva], alfas)
            loc = np.isin(pos, np.flatnonzero(iva))
            for h, p in preds.items():
                curvas[h].append(mae_vaca(vacas[iva], y[iva], p, agregacion))
                oof_h[h][loc] = p
        pesos, media, se = elegir_alfas(alfas, curvas, regla)
        preds = ridge_familia(E[tr], y[tr], pesos_vaca(vacas[tr]), E[ev], list(pesos))
        pred = sum(wt * preds[h] for h, wt in pesos.items())
        oof = sum(wt * oof_h[h] for h, wt in pesos.items())
        info = {"alfas": {f"{h:.4g}": round(wt, 4) for h, wt in pesos.items()},
                "mae_interno": float(np.mean([media[h] for h in pesos])), "se_interno": se}
        return pred, oof, info
    return correr


def _adelante(p, X, rng=None, dropout=0.0):
    z = X @ p["W1"] + p["b1"]
    a = np.maximum(z, 0.0)
    mascara = None
    if rng is not None and dropout > 0:
        mascara = (rng.random(a.shape) >= dropout).astype(np.float32) / (1.0 - dropout)
        a = a * mascara
    return (a @ p["W2"])[:, 0] + p["b2"][0], z, a, mascara


def predecir_mlp(p, X):
    return np.clip(_adelante(p, (X - p["mu"]) / p["sd"])[0].astype(np.float64) + p["centro"],
                   config.BCS_MIN, config.BCS_MAX)


def ajustar_mlp(X, y, w, perdida, semilla, hp, epocas=None, val=None, agregacion="media"):
    """Con val=(Xv, yv, vacas_v): paro temprano por MAE por vaca; si no, `epocas` fijas. Devuelve (params, época)."""
    rng = np.random.default_rng(semilla)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    d, h = X.shape[1], hp["ocultas"]
    centro = float(np.average(y, weights=w))
    p = {"W1": rng.normal(0, np.sqrt(2.0 / d), (d, h)).astype(np.float32), "b1": np.zeros(h, np.float32),
         "W2": np.zeros((h, 1), np.float32), "b2": np.zeros(1, np.float32)}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    yc, w = (y - centro).astype(np.float32), w.astype(np.float32)
    beta, paso = hp["huber_beta"], 0
    mejor = {"mae": np.inf, "epoca": 0, "p": None}
    total = epocas if val is None else hp["epocas_max"]
    for epoca in range(1, total + 1):
        orden = rng.permutation(len(y))
        for i in range(0, len(y), hp["lote"]):
            b = orden[i: i + hp["lote"]]
            out, z, a, mascara = _adelante(p, Xs[b], rng, hp["dropout"])
            err = out - yc[b]
            g = np.sign(err) if perdida == "mae" else np.where(np.abs(err) < beta, err / beta, np.sign(err))
            g = (g * w[b] / len(b)).astype(np.float32)[:, None]
            da = (g @ p["W2"].T) * (z > 0)
            if mascara is not None:
                da = da * mascara
            grads = {"W1": Xs[b].T @ da, "b1": da.sum(0), "W2": a.T @ g, "b2": g.sum(0)}
            paso += 1
            for k in p:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v2[k] = 0.999 * v2[k] + 0.001 * grads[k] ** 2
                upd = (m[k] / (1 - 0.9 ** paso)) / (np.sqrt(v2[k] / (1 - 0.999 ** paso)) + 1e-8)
                if k.startswith("W"):
                    upd = upd + hp["weight_decay"] * p[k]
                p[k] = (p[k] - hp["lr"] * upd).astype(np.float32)
        if val is not None:
            ajustado = {**p, "mu": mu, "sd": sd, "centro": centro}
            mae = mae_vaca(val[2], val[1], predecir_mlp(ajustado, val[0]), agregacion)
            if mae < mejor["mae"] - 1e-4:
                mejor = {"mae": mae, "epoca": epoca, "p": {k: np.copy(v) if isinstance(v, np.ndarray) else v
                                                           for k, v in ajustado.items()}}
            if epoca - mejor["epoca"] >= hp["paciencia"]:
                break
    if val is None:
        return {**p, "mu": mu, "sd": sd, "centro": centro}, total
    return mejor["p"], mejor["epoca"]


def modelo_mlp(rejilla=MLP_REJILLA, semillas=SEMILLAS_MLP, hp_base=MLP_HP):
    """Configuración y épocas por CV interna (paro temprano en el pliegue interno, promedio de semillas);
    final = reajuste en todo el TRAIN con la media de las mejores épocas, promedio de semillas."""
    def correr(E, y, vacas, fold, tr, ev, agregacion="media"):
        internos = sorted(set(fold[tr]))
        pos = np.flatnonzero(tr)
        res = []
        for cfg in rejilla:
            hp = {**hp_base, **{k: v for k, v in cfg.items() if k != "perdida"}}
            maes, epocas, oof = [], [], np.full(tr.sum(), np.nan)
            for j in internos:
                itr, iva = tr & (fold != j), tr & (fold == j)
                w = pesos_vaca(vacas[itr])
                ps = []
                for s in semillas:
                    par, ep = ajustar_mlp(E[itr], y[itr], w, cfg["perdida"], s, hp, val=(E[iva], y[iva], vacas[iva]),
                                          agregacion=agregacion)
                    ps.append(predecir_mlp(par, E[iva]))
                    epocas.append(ep)
                p = np.mean(ps, 0)
                maes.append(mae_vaca(vacas[iva], y[iva], p, agregacion))
                oof[np.isin(pos, np.flatnonzero(iva))] = p
            res.append((float(np.mean(maes)), cfg, hp, max(1, int(round(np.mean(epocas)))), oof))
        mae, cfg, hp, ep, oof = min(res, key=lambda r: r[0])
        w = pesos_vaca(vacas[tr])
        pred = np.mean([predecir_mlp(ajustar_mlp(E[tr], y[tr], w, cfg["perdida"], s, hp, epocas=ep)[0], E[ev])
                        for s in semillas], 0)
        return pred, oof, {"mlp": {**cfg, "epocas": ep}, "mae_interno": mae}
    return correr


def isotonica(x, y):
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o].astype(float)
    vals, pesos, cuenta = [], [], []
    for v in ys:
        vals.append(v); pesos.append(1.0); cuenta.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-12:
            v2, w2, c2 = vals.pop(), pesos.pop(), cuenta.pop()
            vals[-1] = (vals[-1] * pesos[-1] + v2 * w2) / (pesos[-1] + w2)
            pesos[-1] += w2
            cuenta[-1] += c2
    ajustado = np.repeat(vals, cuenta)
    ux, idx = np.unique(xs, return_index=True)
    return ux, np.array([ajustado[i:j].mean() for i, j in zip(idx, list(idx[1:]) + [len(xs)])])


def ajustar_pendiente(p, y):
    """p' = m + b (p - m), m = media de p; b de PENDIENTES maximiza % dentro de ±0.25 (empate: b menor)."""
    m = float(np.mean(p))
    mejor = max((dentro(m + b * (p - m) - y) / len(y), -b, b) for b in PENDIENTES)
    return m, mejor[2], mejor[0]


def ajustar_estiramiento(metodo, p, y):
    if metodo == "afin":
        b, a = np.polyfit(p, y, 1)
        return {"a": float(a), "b": float(b)}, lambda q: a + b * q
    if metodo == "inverso":
        d, c = np.polyfit(y, p, 1)
        d = max(d, 0.05)
        return {"a": float(-c / d), "b": float(1 / d)}, lambda q: (q - c) / d
    if metodo == "isotonico":
        ux, uy = isotonica(p, y)
        return {"nodos": int(len(ux))}, lambda q: np.interp(q, ux, uy)
    if metodo == "pendiente":
        m, b, w025 = ajustar_pendiente(p, y)
        return {"m": m, "b": b, "dentro_025_interno": w025}, lambda q: m + b * (q - m)
    raise ValueError(metodo)


def aplicar_estiramiento(metodo, vacas, y, tr, oof, ev, pred, agregacion="media"):
    if metodo in (None, "ninguno"):
        return aplicar_a_vaca(lambda q: q, vacas[ev], pred, agregacion), None
    v = por_vaca(pd.DataFrame({"id_vaca": vacas[tr], "bcs": y[tr], "pred": oof}), agregacion)
    params, fn = ajustar_estiramiento(metodo, v["pred"].to_numpy(), v["bcs"].to_numpy())
    return aplicar_a_vaca(fn, vacas[ev], pred, agregacion), params


def funcion_base(nombre, mlp_rejilla=MLP_REJILLA):
    if nombre == "mlp":
        return modelo_mlp(mlp_rejilla)
    familia, regla = nombre.split("_", 1)
    if familia != "ridge":
        raise ValueError(nombre)
    return modelo_ridge(regla)


def correr_recetas(E, y, vacas, fold, tr, ev, bases, estiramientos, agregacion="media", mlp_rejilla=MLP_REJILLA):
    salida = {}
    for b in bases:
        pred, oof, info = funcion_base(b, mlp_rejilla)(E, y, vacas, fold, tr, ev, agregacion)
        if np.isnan(oof).any():
            raise ValueError(f"{b}: faltan predicciones fuera de pliegue")
        for s in estiramientos:
            p, params = aplicar_estiramiento(s, vacas, y, tr, oof, ev, pred, agregacion)
            salida[(b, s)] = (p, {**info, "estiramiento": params})
    return salida


def elegir(candidatos, puntajes):
    """Más vacas ±0.25; luego menor MAE; luego el orden de la lista."""
    return min(candidatos, key=lambda c: (-puntajes[c]["n025"], round(puntajes[c]["mae"], 9), candidatos.index(c)))


def seleccion_anidada(E, y, vacas, fold, tr, ev, bases=BASES, estiramientos=ESTIRAMIENTOS, agregacion="media",
                      mlp_rejilla=MLP_REJILLA):
    """Elige la receta con CV anidada sobre los pliegues del TRAIN y predice las fotos `ev` una vez.
    Devuelve (pred por foto de ev, receta elegida, info, puntajes internos)."""
    internos = sorted(set(fold[tr]))
    if len(internos) < 2:
        raise ValueError("la selección anidada necesita al menos 2 pliegues en el TRAIN")
    errores = {}
    for j in internos:
        mtr, mev = tr & (fold != j), tr & (fold == j)
        for r, (p, _) in correr_recetas(E, y, vacas, fold, mtr, mev, bases, estiramientos, agregacion,
                                        mlp_rejilla).items():
            errores.setdefault(r, []).append(errores_vaca(vacas[mev], y[mev], p, agregacion))
    puntajes = {}
    for r, errs in errores.items():
        e = np.concatenate(errs)
        puntajes[r] = {"n025": dentro(e), "n": int(e.size), "mae": float(np.abs(e).mean())}
    candidatos = [(b, s) for b in bases for s in estiramientos]
    receta = elegir(candidatos, puntajes)
    final = correr_recetas(E, y, vacas, fold, tr, ev, [receta[0]], [receta[1]], agregacion, mlp_rejilla)
    pred, info = final[receta]
    return pred, receta, info, {f"{b}|{s}": v for (b, s), v in puntajes.items()}
