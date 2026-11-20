#!/usr/bin/env python3
"""
comparar_tiempos.py - Corre indice_pandas.py e indice_spark.py varias veces, mide tiempo, CPU y memoria
de cada corrida y verifica que las dos versiones generen exactamente las mismas tablas.

Cada corrida es un proceso nuevo, así que Spark arranca su JVM cada vez, igual que en un uso real.
Si se da --referencia (la carpeta con los CSV del dataset que está en S3), también se verifica que
el resultado sea igual al dataset del bucket.

Uso:
    python3 comparar_tiempos.py --entrada entrada --salida salida
    python3 comparar_tiempos.py --entrada entrada --salida salida --referencia dataset_csv
    python3 comparar_tiempos.py --entrada entrada --salida salida --master spark://<ip-del-master>:7077

Genera en --salida:
    pandas/ y spark/      tablas de cada versión
    tiempos.csv           una fila por corrida, con tiempo, CPU y memoria
    verificacion.json     si pandas y Spark coinciden, y si coinciden con la referencia
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from medicion import Medidor, entorno
from reglas import ENTRADAS, SALIDAS, NUMERICAS, LLAVES

AQUI = Path(__file__).resolve().parent


def correr(script, entrada, salida, extra=()):
    cmd = [sys.executable, str(AQUI / script), "--entrada", str(entrada), "--salida", str(salida), *extra]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=AQUI)
    with Medidor(pid=p.pid) as m:
        salida_std, error_std = p.communicate()
    if p.returncode != 0:
        print(salida_std[-3000:])
        print(error_std[-3000:])
        sys.exit(f"\nFalló {script}. Arriba está el error.")
    linea = [x for x in salida_std.splitlines() if x.startswith("TIEMPOS_JSON ")]
    t = json.loads(linea[-1].split(" ", 1)[1]) if linea else {}
    return {"t_total_proceso": m.resultado["segundos"], **t, "cpu_s": m.resultado["cpu_s"],
            "cpu_pct": m.resultado["cpu_pct"], "mem_max_mb": m.resultado["mem_max_mb"]}


def leer(ruta):
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def comparar(dir_a, dir_b, nombre_a="pandas", nombre_b="spark"):
    """Regresa la lista de diferencias entre las tablas de dos carpetas (vacía si todo coincide)."""
    difs = []
    for nombre in SALIDAS:
        a, b = leer(dir_a / nombre), leer(dir_b / nombre)
        if list(a.columns) != list(b.columns):
            difs.append(f"{nombre}: columnas distintas {list(a.columns)} vs {list(b.columns)}")
            continue
        if len(a) != len(b):
            difs.append(f"{nombre}: {len(a)} filas en {nombre_a} vs {len(b)} en {nombre_b}")
            continue
        llave = LLAVES[nombre]
        a = a.sort_values(llave).reset_index(drop=True)
        b = b.sort_values(llave).reset_index(drop=True)
        for c in a.columns:
            if c in NUMERICAS:
                na, nb = pd.to_numeric(a[c], errors="coerce"), pd.to_numeric(b[c], errors="coerce")
                malas = ~(((na - nb).abs() < 1e-9) | (na.isna() & nb.isna()))
            else:
                malas = a[c] != b[c]
            if malas.any():
                difs.append(f"{nombre}, columna '{c}': {int(malas.sum())} diferencias")
    return difs


def main():
    ap = argparse.ArgumentParser(description="pandas contra PySpark: tiempo, CPU, memoria y verificación.")
    ap.add_argument("--entrada", default="entrada")
    ap.add_argument("--salida", default="salida")
    ap.add_argument("--repeticiones", type=int, default=5)
    ap.add_argument("--master", default="local[*]", help="master de Spark (local[*] o spark://<ip>:7077)")
    ap.add_argument("--referencia", default=None, help="carpeta con los CSV del dataset del bucket")
    a = ap.parse_args()

    entrada, salida = Path(a.entrada).resolve(), Path(a.salida).resolve()
    faltan = [n for n in ENTRADAS if not (entrada / n).is_file()]
    if faltan:
        sys.exit(f"Faltan en {entrada}: {faltan}")
    salida.mkdir(parents=True, exist_ok=True)

    filas = []
    for impl, script, extra in (("pandas", "indice_pandas.py", ()),
                                ("spark", "indice_spark.py", ("--master", a.master))):
        for rep in range(1, a.repeticiones + 1):
            print(f"{impl}: corrida {rep}/{a.repeticiones} ...", flush=True)
            t = correr(script, entrada, salida / impl, extra)
            filas.append({"implementacion": impl, "master": a.master if impl == "spark" else "",
                          "repeticion": rep, **t})

    cols = ["implementacion", "master", "repeticion", "t_total_proceso", "t_imports", "t_sesion",
            "t_procesamiento", "cpu_s", "cpu_pct", "mem_max_mb"]
    tiempos = pd.DataFrame(filas).reindex(columns=cols)
    tiempos.round(4).to_csv(salida / "tiempos.csv", index=False)

    difs = comparar(salida / "pandas", salida / "spark")
    difs_ref = comparar(salida / "pandas", Path(a.referencia), "pandas", "bucket") if a.referencia else None
    info = {"iguales": not difs, "diferencias": difs, "repeticiones": a.repeticiones, "master": a.master,
            "igual_a_referencia": (not difs_ref) if difs_ref is not None else None,
            "diferencias_con_referencia": difs_ref, "maquina": {**entorno(), "python": platform.python_version()},
            "fecha_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    (salida / "verificacion.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n==== Mediana de {a.repeticiones} corridas ====")
    resumen = tiempos.groupby("implementacion")[cols[3:]].median().round(3)
    print(resumen.to_string())
    mp, ms = resumen.loc["pandas", "t_total_proceso"], resumen.loc["spark", "t_total_proceso"]
    print(f"Spark / pandas en tiempo total: {ms / mp:.1f} veces")

    print("\n==== Verificación ====")
    print("pandas y Spark generaron exactamente las mismas tablas." if not difs else
          "pandas y Spark NO coinciden:\n  - " + "\n  - ".join(difs))
    if difs_ref is not None:
        print("El resultado es igual al dataset del bucket." if not difs_ref else
              "El resultado NO es igual al dataset del bucket:\n  - " + "\n  - ".join(difs_ref))
    if difs or difs_ref:
        sys.exit(1)


if __name__ == "__main__":
    main()
