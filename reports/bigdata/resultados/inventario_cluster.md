# Inventario del clúster Spark

Clúster standalone de Spark 3.5.3 (PySpark 3.5.3, Python 3.10.22, Java 17) por Tailscale, con 3 máquinas. El master y el driver corren en la MBP; los workers son la PC (dentro de WSL) y la MacBook Air.

| Máquina | Rol | Núcleos de la máquina | RAM de la máquina | Asignado en Spark |
|---|---|---|---|---|
| MacBook Pro (bixku-mbp) | master y driver | 8 | 16 GB | driver: 3 GB |
| PC (pc-bixku, WSL) | worker | 12 | 24 GB (los que ve WSL) | 8 núcleos y 16 GB |
| MacBook Air (bixku-mba) | worker | 8 | 8 GB | 4 núcleos y 4 GB |

Los workers suman 12 núcleos y 20 GB para los jobs. La MBP no ejecuta tareas, solo coordina y recibe las tablas finales.

Las tres máquinas se hablan solo por la interfaz de Tailscale (master en el puerto 7077, UI en el 8080).

Los tiempos están en `tiempos_cluster.csv` y la verificación en `verificacion_cluster.json`. En `verificacion_cluster.json`, el campo `maquina` describe solo la MBP, donde corrió el driver.
