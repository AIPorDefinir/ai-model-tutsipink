# Resultados

Cada archivo de esta carpeta lo genera un notebook al terminar de correr y se sube tal cual, sin editarlo. Ninguno tiene IDs de vacas ni datos de las fotos.

| Archivo | Lo genera | Contenido |
|---|---|---|
| `carga_resumen.json` | `src/notebooks/bigdata/preparacion_datos.ipynb` (sección de carga) | Comparación del dataset generado con el del bucket: archivos iguales, distintos, faltantes y sobrantes |
| `resumen_pliegues.csv` | `src/notebooks/bigdata/validacion_cruzada.ipynb` | Vacas, fotos por vista, BCS y evaluaciones de cada pliegue |
| `rotaciones_modelo_base.csv` | `src/notebooks/bigdata/validacion_cruzada.ipynb` | Conjuntos de cada rotación y métricas del modelo base |
| `tiempos_recursos.csv` | `src/notebooks/bigdata/medicion_recursos.ipynb` | Tiempo, CPU y memoria de cada etapa y de cada corrida |
| `entorno.json` | `src/notebooks/bigdata/medicion_recursos.ipynb` | Núcleos, memoria y versiones de la máquina donde se midió |
