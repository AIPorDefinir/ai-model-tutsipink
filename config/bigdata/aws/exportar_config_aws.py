#!/usr/bin/env python3
"""
exportar_config_aws.py - Exporta a JSON la configuración de AWS del reto, solo leyendo.

La configuración del bucket, de IAM, de CloudTrail y de CloudWatch se hizo desde la consola de AWS. Este script
la lee y la guarda en archivos JSON para que quede como evidencia en el repositorio. Solo usa llamadas de lectura
(Get, List y Describe), así que no cambia nada en la cuenta.

Antes de guardar, reemplaza el ID de la cuenta por <ID_CUENTA>, el nombre del bucket por <BUCKET> y los nombres
de los usuarios de IAM por usuario_1, usuario_2, etc. Aun así, revisa los archivos antes de subirlos.

Necesita una identidad con permisos de lectura sobre IAM, S3, CloudTrail, CloudWatch y Budgets (por ejemplo la
de quien configuró la cuenta). Si alguna llamada no tiene permiso, se anota el error y el script sigue.

Uso:
    pip install boto3
    python3 exportar_config_aws.py --bucket <nombre-del-bucket> --perfil <perfil-de-aws-cli> --salida exportado
"""
import argparse
import json
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

REGION = "mx-central-1"
REGION_FACTURACION = "us-east-1"  # las alarmas de facturación y Budgets viven aquí


def intentar(funcion, *args, **kwargs):
    try:
        r = funcion(*args, **kwargs)
        r.pop("ResponseMetadata", None)
        return r
    except ClientError as e:
        return {"error": e.response.get("Error", {}).get("Code", "ClientError")}
    except BotoCoreError as e:
        return {"error": type(e).__name__}


def exportar_s3(s3, bucket):
    datos = {}
    for nombre, funcion in (("ubicacion", s3.get_bucket_location), ("politica", s3.get_bucket_policy),
                            ("cifrado", s3.get_bucket_encryption), ("bloqueo_acceso_publico", s3.get_public_access_block),
                            ("propiedad_de_objetos", s3.get_bucket_ownership_controls),
                            ("versionado", s3.get_bucket_versioning), ("registro_de_accesos", s3.get_bucket_logging),
                            ("ciclo_de_vida", s3.get_bucket_lifecycle_configuration)):
        datos[nombre] = intentar(funcion, Bucket=bucket)
    if isinstance(datos["politica"].get("Policy"), str):
        datos["politica"]["Policy"] = json.loads(datos["politica"]["Policy"])
    return datos


def exportar_iam(iam):
    datos = {"usuarios": [], "grupos": [], "politicas_administradas_por_el_equipo": {},
             "politica_de_contrasenas": intentar(iam.get_account_password_policy)}
    propias = set()
    usuarios = intentar(iam.list_users).get("Users", [])
    for u in usuarios:
        n = u["UserName"]
        adjuntas = intentar(iam.list_attached_user_policies, UserName=n).get("AttachedPolicies", [])
        propias |= {p["PolicyArn"] for p in adjuntas if ":aws:policy/" not in p["PolicyArn"]}
        datos["usuarios"].append({
            "usuario": n, "creado": u.get("CreateDate"),
            "dispositivos_mfa": len(intentar(iam.list_mfa_devices, UserName=n).get("MFADevices", [])),
            "grupos": [g["GroupName"] for g in intentar(iam.list_groups_for_user, UserName=n).get("Groups", [])],
            "politicas_adjuntas": [p["PolicyName"] for p in adjuntas],
            "politicas_en_linea": {p: intentar(iam.get_user_policy, UserName=n, PolicyName=p).get("PolicyDocument")
                                   for p in intentar(iam.list_user_policies, UserName=n).get("PolicyNames", [])},
        })
    for g in intentar(iam.list_groups).get("Groups", []):
        n = g["GroupName"]
        adjuntas = intentar(iam.list_attached_group_policies, GroupName=n).get("AttachedPolicies", [])
        propias |= {p["PolicyArn"] for p in adjuntas if ":aws:policy/" not in p["PolicyArn"]}
        datos["grupos"].append({
            "grupo": n, "politicas_adjuntas": [p["PolicyName"] for p in adjuntas],
            "politicas_en_linea": {p: intentar(iam.get_group_policy, GroupName=n, PolicyName=p).get("PolicyDocument")
                                   for p in intentar(iam.list_group_policies, GroupName=n).get("PolicyNames", [])},
        })
    for arn in sorted(propias):
        pol = intentar(iam.get_policy, PolicyArn=arn).get("Policy", {})
        version = pol.get("DefaultVersionId")
        doc = intentar(iam.get_policy_version, PolicyArn=arn, VersionId=version).get("PolicyVersion", {}) if version else {}
        datos["politicas_administradas_por_el_equipo"][pol.get("PolicyName", arn)] = doc.get("Document")
    return datos, [u["UserName"] for u in usuarios]


def exportar_cloudtrail(sesion):
    datos = {}
    for region in (REGION, REGION_FACTURACION):
        ct = sesion.client("cloudtrail", region_name=region)
        trails = intentar(ct.describe_trails, includeShadowTrails=False).get("trailList", [])
        datos[region] = [{"trail": t, "estado": intentar(ct.get_trail_status, Name=t["TrailARN"]),
                          "selectores_de_eventos": intentar(ct.get_event_selectors, TrailName=t["TrailARN"])}
                         for t in trails]
    return datos


def exportar_cloudwatch(sesion):
    datos = {}
    for region in (REGION, REGION_FACTURACION):
        cw = sesion.client("cloudwatch", region_name=region)
        alarmas = intentar(cw.describe_alarms).get("MetricAlarms", [])
        datos[region] = [{k: a.get(k) for k in ("AlarmName", "MetricName", "Namespace", "Statistic", "Period",
                                                  "Threshold", "ComparisonOperator", "Dimensions", "ActionsEnabled",
                                                  "AlarmActions")} for a in alarmas]
    return datos


def anonimizar(obj, reemplazos):
    if isinstance(obj, dict):
        return {anonimizar(k, reemplazos): anonimizar(v, reemplazos) for k, v in obj.items()}
    if isinstance(obj, list):
        return [anonimizar(v, reemplazos) for v in obj]
    if isinstance(obj, str):
        for viejo, nuevo in reemplazos:
            obj = obj.replace(viejo, nuevo)
    return obj


def main():
    ap = argparse.ArgumentParser(description="Exporta la configuración de AWS del reto (solo lectura).")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--perfil", default=None, help="perfil de AWS CLI; si no se da, se usan las variables de entorno")
    ap.add_argument("--salida", default="exportado")
    a = ap.parse_args()

    sesion = boto3.Session(profile_name=a.perfil, region_name=REGION)
    cuenta = sesion.client("sts").get_caller_identity()["Account"]
    iam, usuarios = exportar_iam(sesion.client("iam"))
    archivos = {
        "s3_bucket.json": exportar_s3(sesion.client("s3"), a.bucket),
        "iam.json": iam,
        "cloudtrail.json": exportar_cloudtrail(sesion),
        "cloudwatch_alarmas.json": exportar_cloudwatch(sesion),
        "budgets.json": intentar(sesion.client("budgets", region_name=REGION_FACTURACION).describe_budgets,
                                 AccountId=cuenta),
    }

    reemplazos = [(a.bucket, "<BUCKET>"), (cuenta, "<ID_CUENTA>")]
    reemplazos += [(u, f"usuario_{i}") for i, u in sorted(enumerate(usuarios, 1), key=lambda x: -len(x[1]))]
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, datos in archivos.items():
        limpio = anonimizar(json.loads(json.dumps(datos, default=str)), reemplazos)
        (salida / nombre).write_text(json.dumps(limpio, ensure_ascii=False, indent=2), encoding="utf-8")
        errores = json.dumps(limpio).count('"error"')
        print(f"{salida / nombre}" + (f"  ({errores} consultas con error, ver el campo error)" if errores else ""))
    print("Revisa los archivos antes de subirlos: no deben tener llaves, correos ni nombres de personas.")


if __name__ == "__main__":
    main()
