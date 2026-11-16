"""Modelo 3: ajuste fino completo de EVA02 (PyTorch + timm, GPU), ensamble de 5 semillas combinado por vaca."""
import json
import math
import time

import numpy as np
import pandas as pd

from . import config, protocol
from .metrics import dentro, pesos_vaca, por_vaca

RECETA = dict(modelo="eva02_base_patch14_224.mim_in22k", tamano=(392, 294), lrs=[1e-5, 3e-5], epocas=15,
              lr_cabeza=1e-3, weight_decay=0.05, lote=8, dropout=0.3, huber_beta=0.25, checkpointing=True)
SEMILLAS = [2026, 2027, 2028, 2029, 2030]


def _red(receta, centro):
    import timm
    import torch.nn as nn

    class Red(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = timm.create_model(receta["modelo"], pretrained=True, num_classes=0, dynamic_img_size=True)
            if receta.get("checkpointing"):
                self.bb.set_grad_checkpointing(True)
            self.npre = getattr(self.bb, "num_prefix_tokens", 1)
            self.drop = nn.Dropout(receta["dropout"])
            self.cabeza = nn.Linear(self.bb.num_features * 2, 1)
            nn.init.zeros_(self.cabeza.weight)
            nn.init.constant_(self.cabeza.bias, centro)

        def forward(self, x):
            import torch

            t = self.bb.forward_features(x)
            f = torch.cat([t[:, 0], t[:, self.npre:].mean(1)], 1)
            return self.cabeza(self.drop(f.float())).squeeze(1)

    return Red()


def _preparar(xb, tamano, media, desv, entrenar, g):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.v2.functional as TF

    x = xb.permute(0, 3, 1, 2).float() / 255.0
    if entrenar:
        salida = []
        for im in x:
            if torch.rand(1, generator=g).item() < 0.5:
                im = torch.flip(im, dims=[2])
            im = TF.rotate(im, (torch.rand(1, generator=g).item() * 2 - 1) * 10, interpolation=TF.InterpolationMode.BILINEAR)
            s = 0.85 + 0.15 * torch.rand(1, generator=g).item()
            H, W = im.shape[1:]
            h, w = int(H * math.sqrt(s)), int(W * math.sqrt(s))
            t0 = int(torch.randint(0, H - h + 1, (1,), generator=g))
            l0 = int(torch.randint(0, W - w + 1, (1,), generator=g))
            im = im[:, t0:t0 + h, l0:l0 + w]
            im = TF.adjust_brightness(im, 1 + (torch.rand(1, generator=g).item() * 2 - 1) * 0.1)
            im = TF.adjust_contrast(im, 1 + (torch.rand(1, generator=g).item() * 2 - 1) * 0.1)
            salida.append(F.interpolate(im[None], size=tamano, mode="bilinear", antialias=True, align_corners=False)[0])
        x = torch.stack(salida)
    else:
        x = F.interpolate(x, size=tamano, mode="bilinear", antialias=True, align_corners=False)
    return (x - media) / desv


def entrenar_evaluar(R, X, y, w, idx_ajuste, idx_eval, lr, epocas, semilla, al_terminar_epoca=None):
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch no ve la GPU: el ajuste fino de EVA02 necesita GPU")
    dev = torch.device("cuda")
    torch.manual_seed(semilla)
    g = torch.Generator().manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    red = _red(R, float(np.average(y[idx_ajuste], weights=w[idx_ajuste]))).to(dev)
    cfg = red.bb.pretrained_cfg
    media = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    desv = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    opt = torch.optim.AdamW([{"params": red.bb.parameters(), "lr": lr},
                             {"params": red.cabeza.parameters(), "lr": R["lr_cabeza"]}], weight_decay=R["weight_decay"])
    base = [lr, R["lr_cabeza"]]
    pasos_epoca = math.ceil(len(idx_ajuste) / R["lote"])
    total = pasos_epoca * epocas
    escalador = torch.amp.GradScaler("cuda")
    wt = torch.tensor(w / w[idx_ajuste].mean(), dtype=torch.float32)
    paso = 0

    def predecir(idx):
        red.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(idx), 16):
                x = _preparar(torch.from_numpy(X[idx[i:i + 16]]).to(dev), R["tamano"], media, desv, False, g)
                out.append(((red(x) + red(torch.flip(x, dims=[3]))) / 2).float().cpu().numpy())
        red.train()
        return np.clip(np.concatenate(out), config.BCS_MIN, config.BCS_MAX)

    red.train()
    for ep in range(1, epocas + 1):
        orden = rng.permutation(idx_ajuste)
        for i in range(0, len(orden), R["lote"]):
            b = orden[i:i + R["lote"]]
            f = (paso + 1) / pasos_epoca if paso < pasos_epoca else \
                0.5 * (1 + math.cos(math.pi * (paso - pasos_epoca) / max(1, total - pasos_epoca)))
            for pg, lr0 in zip(opt.param_groups, base):
                pg["lr"] = lr0 * f
            x = _preparar(torch.from_numpy(X[b]).to(dev), R["tamano"], media, desv, True, g)
            yt = torch.tensor(y[b], dtype=torch.float32, device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                p = red(x)
            wb = wt[b].to(dev)
            perdida = (F.smooth_l1_loss(p.float(), yt, beta=R["huber_beta"], reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            paso += 1
        if al_terminar_epoca is not None:
            al_terminar_epoca(ep, predecir(idx_eval))
    pred = predecir(idx_eval)
    del red, opt
    torch.cuda.empty_cache()
    return pred


def elegir_lr_epoca(tabla):
    """tabla: filas {lr, epoca, mae}. Menor MAE fuera de pliegue; empates: menos épocas, lr menor."""
    return min(tabla, key=lambda r: (round(r["mae"], 6), r["epoca"], r["lr"]))


def tarea_miembro(cfg, k, semilla):
    """Un miembro del ensamble para el pliegue externo k. Guarda su parte en config.AJUSTE_FINO y devuelve sus
    predicciones por vaca (mediana de fotos) con salida = miembro_<semilla>."""
    from .network import decodificar
    from .tasks import filas_vacas
    from .train import leer_fotos

    R = {**RECETA, **cfg.get("receta", {})}
    R["tamano"] = tuple(R["tamano"])
    protocolo = cfg["protocolo"]
    destino = config.AJUSTE_FINO / cfg["nombre"] / f"k{k}_s{semilla}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fotos = leer_fotos(cfg["entrada"])
    X = decodificar(fotos["jpg"].tolist())
    y, vacas, fold = fotos["bcs"].to_numpy(float), fotos["id_vaca"].to_numpy(), fotos["fold"].to_numpy(int)
    w = pesos_vaca(vacas)
    ev, tr = protocol.particion(fold, k, protocolo)
    protocol.revisar(vacas, ev, tr)
    t0 = time.time()
    if not destino.exists():
        internos = sorted(set(fold[tr].tolist()))
        v_tr = por_vaca(pd.DataFrame({"id_vaca": vacas[tr], "bcs": y[tr], "pred": y[tr]}))
        vacas_tr, y_tr = v_tr["id_vaca"].to_numpy(), v_tr["bcs"].to_numpy()
        curvas = {lr: {e: pd.Series(np.nan, index=vacas_tr) for e in range(1, R["epocas"] + 1)} for lr in R["lrs"]}
        pliegue_vaca = pd.Series(fold[tr], index=vacas[tr]).groupby(level=0).first()
        for lr in R["lrs"]:
            for j in internos:
                idx_aj, idx_ev = np.flatnonzero(tr & (fold != j)), np.flatnonzero(tr & (fold == j))

                def al_terminar(ep, p, lr=lr, idx_ev=idx_ev):
                    v = por_vaca(pd.DataFrame({"id_vaca": vacas[idx_ev], "bcs": y[idx_ev], "pred": p}))
                    curvas[lr][ep].loc[v["id_vaca"].to_numpy()] = v["pred"].to_numpy()

                entrenar_evaluar(R, X, y, w, idx_aj, idx_ev, lr, R["epocas"], semilla + k, al_terminar)
        tabla = [{"lr": lr, "epoca": e, "mae": float(np.mean(np.abs(curvas[lr][e].loc[vacas_tr].to_numpy() - y_tr)))}
                 for lr in R["lrs"] for e in range(1, R["epocas"] + 1)]
        mejor = elegir_lr_epoca(tabla)
        oof = curvas[mejor["lr"]][mejor["epoca"]].loc[vacas_tr]
        idx_ev_ext = np.flatnonzero(ev)
        pred = entrenar_evaluar(R, X, y, w, np.flatnonzero(tr), idx_ev_ext, mejor["lr"], mejor["epoca"], semilla + k)
        parte = {"k": k, "semilla": semilla, "eleccion": mejor, "tabla": tabla,
                 "oof": {"id_vaca": oof.index.tolist(), "pred": oof.tolist(),
                         "j": pliegue_vaca.loc[oof.index].astype(int).tolist(), "bcs": y_tr.tolist()},
                 "eval": {"ruta": fotos["ruta"].to_numpy()[idx_ev_ext].tolist(), "pred": pred.tolist()},
                 "segundos": round(time.time() - t0)}
        tmp = destino.with_suffix(".tmp")
        tmp.write_text(json.dumps(parte))
        tmp.rename(destino)
    parte = json.loads(destino.read_text())
    f_ev = fotos[ev].set_index("ruta").loc[parte["eval"]["ruta"]].reset_index()
    v = por_vaca(f_ev.assign(pred=parte["eval"]["pred"])[["id_vaca", "bcs", "pred"]], "mediana")
    return filas_vacas(v, cfg["nombre"], f"miembro_{semilla}", k, protocol.pliegue_evaluado(k, protocolo), y[tr],
                       {"eleccion": parte["eleccion"], "semilla": semilla})


def decidir_pendiente(oof, y, j):
    """Pendiente anidada sobre los pliegues internos: se usa solo si deja estrictamente más vacas ±0.25."""
    from .heads import ajustar_pendiente

    n_crudo = dentro(oof - y)
    n_anidado = 0
    for jj in np.unique(j):
        m, b, _ = ajustar_pendiente(oof[j != jj], y[j != jj])
        p = np.clip(m + b * (oof[j == jj] - m), config.BCS_MIN, config.BCS_MAX)
        n_anidado += dentro(p - y[j == jj])
    m, b, _ = ajustar_pendiente(oof, y)
    return {"usar": n_anidado > n_crudo, "m": m, "b": b, "n_crudo": n_crudo, "n_anidado": n_anidado}


def combinar_partes(partes, fotos_ev):
    """partes: JSON de los miembros de UN pliegue externo; fotos_ev: ruta, id_vaca, bcs del pliegue evaluado.
    Devuelve (DataFrame id_vaca, bcs, pred, decisión de pendiente)."""
    oof = pd.concat([pd.Series(p["oof"]["pred"], index=p["oof"]["id_vaca"]) for p in partes], axis=1).mean(axis=1)
    ref = partes[0]["oof"]
    j = pd.Series(ref["j"], index=ref["id_vaca"]).loc[oof.index].to_numpy()
    y = pd.Series(ref["bcs"], index=ref["id_vaca"]).loc[oof.index].to_numpy()
    d = decidir_pendiente(oof.to_numpy(), y, j)
    pred = np.mean([pd.Series(p["eval"]["pred"], index=p["eval"]["ruta"]).loc[fotos_ev["ruta"]].to_numpy()
                    for p in partes], axis=0)
    v = por_vaca(fotos_ev.assign(pred=pred)[["id_vaca", "bcs", "pred"]], "mediana")
    if d["usar"]:
        v["pred"] = np.clip(d["m"] + d["b"] * (v["pred"] - d["m"]), config.BCS_MIN, config.BCS_MAX)
    return v, d


def combinar(spark, cfg, destino):
    """Combina los miembros de cada pliegue y agrega la salida 'continua' al Parquet de predicciones del modelo."""
    from .tasks import filas_vacas
    from .train import leer_fotos

    fotos = leer_fotos(cfg["entrada"]).drop(columns="jpg")
    semillas = cfg.get("semillas", SEMILLAS)
    filas = []
    for k in range(config.K):
        partes = [json.loads((config.AJUSTE_FINO / cfg["nombre"] / f"k{k}_s{s}.json").read_text()) for s in semillas]
        ev, tr = protocol.particion(fotos["fold"].to_numpy(), k, cfg["protocolo"])
        v, d = combinar_partes(partes, fotos[ev][["ruta", "id_vaca", "bcs"]])
        filas.append(filas_vacas(v, cfg["nombre"], "continua", k, protocol.pliegue_evaluado(k, cfg["protocolo"]),
                                 fotos["bcs"].to_numpy()[tr], {"pendiente": d, "semillas": semillas}))
    df = spark.createDataFrame(pd.concat(filas, ignore_index=True).drop(columns="salida"))
    df.write.mode("overwrite").parquet(str(destino / "salida=continua"))
