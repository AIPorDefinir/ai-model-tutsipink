#!/usr/bin/env python3
"""
preparar_entradas.py - Arma las entradas de indice_pandas.py e indice_spark.py a partir del dataset del bucket.

El notebook de preparación genera fotos.csv y bitacora.csv en su carpeta intermedios/ de Colab, y esa carpeta
no se guarda en el bucket. Para medir la integración sin repetir toda la preparación, este script reconstruye
esas dos entradas con los CSV del dataset (indice_imagenes.csv y evaluaciones.csv) y toma vistas.csv tal cual.

    fotos.csv     una fila por foto del índice, con los mismos metadatos (no incluye las fotos excluidas,
                  que de todos modos no pasan a la integración)
    bitacora.csv  una fila por evaluación de evaluaciones.csv, en el mismo orden (no incluye las evaluaciones
                  excluidas, que tampoco pasan a la integración)
    vistas.csv    las marcas de vista

Con estas entradas, la integración debe reproducir exactamente el dataset del bucket, y comparar_tiempos.py
lo verifica con --referencia.

Con --multiplicar N se generan entradas N veces más grandes para medir cómo crece el tiempo con el volumen.
Cada copia repite todas las vacas con un ID distinto (el ID original más 1,000,000 × número de copia), así
que es un volumen simulado a partir de los datos reales y sirve solo para medir, no para entrenar.

Uso:
    python3 preparar_entradas.py --dataset dataset_csv --vistas vistas.csv --salida entrada
    python3 preparar_entradas.py --dataset dataset_csv --vistas vistas.csv --salida entrada_x10 --multiplicar 10
"""
import argparse
from pathlib import Path

import pandas as pd


def leer(ruta):
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def multiplicar(fotos, bit, vis, n):
    """Repite las tablas n veces; en la copia r el ID de cada vaca se convierte en ID + r × 1,000,000."""
    def copia(df, r, con_ruta):
        df = df.copy()
        nuevo = (pd.to_numeric(df["id_vaca"]) + r * 1_000_000).astype("int64").astype(str)
        if con_ruta:
            df["archivo_jpg"] = "fotos_jpg/" + nuevo + "/" + df["archivo_jpg"].str.split("/").str[-1]
        df["id_vaca"] = nuevo
        return df
    return (pd.concat([copia(fotos, r, True) for r in range(n)], ignore_index=True),
            pd.concat([copia(bit, r, False) for r in range(n)], ignore_index=True),
            pd.concat([copia(vis, r, True) for r in range(n)], ignore_index=True))


def main():
    ap = argparse.ArgumentParser(description="Entradas de la integración a partir del dataset del bucket.")
    ap.add_argument("--dataset", required=True, help="carpeta con indice_imagenes.csv y evaluaciones.csv")
    ap.add_argument("--vistas", required=True, help="vistas.csv con las marcas de vista")
    ap.add_argument("--salida", default="entrada")
    ap.add_argument("--multiplicar", type=int, default=1, help="repite los datos N veces (solo para medir)")
    a = ap.parse_args()

    dataset, salida = Path(a.dataset), Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    idx = leer(dataset / "indice_imagenes.csv")
    ev = leer(dataset / "evaluaciones.csv")

    fotos = idx[["id_vaca", "archivo_jpg", "archivo_original", "sha256", "fecha_hora", "camara", "ancho", "alto",
                 "profundidad"]].copy()
    fotos.insert(2, "excluida", "no")
    fotos.insert(3, "motivo", "")
    bit = pd.DataFrame({"fila_excel": ev["fila_excel"], "id_vaca": ev["id_vaca"], "hora": ev["hora"],
                        "hora_original": "", "bcs": ev["bcs"], "excluida": "no", "motivo": ""})
    vis = leer(a.vistas)
    if a.multiplicar > 1:
        fotos, bit, vis = multiplicar(fotos, bit, vis, a.multiplicar)

    fotos.to_csv(salida / "fotos.csv", index=False)
    bit.to_csv(salida / "bitacora.csv", index=False)
    vis.to_csv(salida / "vistas.csv", index=False)
    print(f"Entradas en {salida}: fotos.csv ({len(fotos)} fotos), bitacora.csv ({len(bit)} evaluaciones), "
          f"vistas.csv ({len(vis)} marcas)")


if __name__ == "__main__":
    main()
