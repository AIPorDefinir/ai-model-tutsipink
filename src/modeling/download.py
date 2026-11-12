import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from . import config

DESTINO = config.DATA / "dataset_bcs"


def cliente(servicio="s3"):
    import boto3

    config.cargar_env()
    return boto3.Session(region_name=config.REGION).client(servicio)


def identidad():
    i = cliente("sts").get_caller_identity()
    print(f"Cuenta {i['Account']}, identidad {i['Arn']}")
    if "assumed-role" in i["Arn"] or "sso" in i["Arn"].lower():
        raise SystemExit("La identidad es un rol (¿SSO?), no el usuario IAM de las llaves del .env")
    return i


def listar(s3, prefijo):
    for pagina in s3.get_paginator("list_objects_v2").paginate(Bucket=config.bucket(), Prefix=prefijo):
        yield from pagina.get("Contents", [])


def buscar(s3):
    encontrados = [o["Key"] for o in listar(s3, "") if o["Key"].endswith("indice_imagenes.csv")]
    for k in encontrados:
        print(k)
    return encontrados


def descargar(s3, prefijo=config.PREFIJO_DATASET, destino=DESTINO, hilos=16):
    bucket = config.bucket()
    prefijo = prefijo.strip("/") + "/"
    objetos = [o for o in listar(s3, prefijo) if not o["Key"].endswith("/")]
    if not objetos:
        raise SystemExit(f"No hay archivos en s3://{bucket}/{prefijo}. Prueba con --buscar.")

    def bajar(o):
        archivo = destino / o["Key"][len(prefijo):]
        if archivo.exists() and archivo.stat().st_size == o["Size"]:
            return 0
        archivo.parent.mkdir(parents=True, exist_ok=True)
        tmp = archivo.with_name(archivo.name + ".tmp")
        s3.download_file(bucket, o["Key"], str(tmp))
        tmp.rename(archivo)
        return o["Size"]

    t0 = time.time()
    with ThreadPoolExecutor(hilos) as ex:
        bajados = list(ex.map(bajar, objetos))
    total = sum(o["Size"] for o in objetos)
    print(
        f"s3://{bucket}/{prefijo} -> {destino}: {len(objetos)} archivos ({total / 1e9:.2f} GB), "
        f"{sum(1 for b in bajados if b)} descargados ({sum(bajados) / 1e9:.2f} GB) en {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--identidad", action="store_true")
    p.add_argument("--buscar", action="store_true")
    p.add_argument("--prefijo", default=config.PREFIJO_DATASET)
    a = p.parse_args()
    identidad()
    s3 = cliente()
    if a.buscar:
        buscar(s3)
    elif not a.identidad:
        descargar(s3, a.prefijo)
