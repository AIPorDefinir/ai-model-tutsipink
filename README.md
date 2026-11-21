# Reto de la concentración "Inteligencia artificial avanzada para la ciencia de datos"

Proyecto del reto con el CAETEC: estimar la condición corporal (BCS) de las vacas a partir de fotos. El proyecto usa la metodología CRISP-DM, y en este repositorio también están los módulos de Big Data y de Privacidad y seguridad de datos.

<img src="docs/img/crisp-dm.png" alt="Diagrama del proceso CRISP-DM" width="400">

<sub>Diagrama: Kenneth Jensen, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), vía [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:CRISP-DM_Process_Diagram.png).</sub>

Plan del proyecto: [PVG](https://aipordefinir.github.io/ai-model-tutsipink/)

## Contenido

- [Estructura del repositorio](#estructura-del-repositorio)
- [CRISP-DM](#crisp-dm)
- [Módulo de Big Data](#módulo-de-big-data)
- [Módulo de Privacidad y seguridad de datos](#módulo-de-privacidad-y-seguridad-de-datos)
- [Contributors](#contributors)

## Estructura del repositorio

Las fuentes (código, notebooks y LaTeX) están en `src/` y los documentos que se entregan, en `reports/`. Dentro de cada carpeta hay una subcarpeta por fase de CRISP-DM o por módulo.

```
├── README.md
├── LICENSE
├── requirements.txt                   paquetes del modelado
├── config/
│   └── bigdata/                       paquetes y configuración exportada de AWS del módulo de Big Data
├── docs/                              PVG e imágenes del README
├── reports/                           documentos que se entregan (PDF)
│   ├── 01-business-understanding/
│   ├── 02-data-understanding/
│   ├── 03-data-preparation/
│   ├── 04-modeling/
│   ├── bigdata/                       reporte de Big Data y resultados de los notebooks
│   └── privacidad-datos/              reportes individuales de privacidad
└── src/                               fuentes
    ├── modeling/                      código del modelado
    ├── bigdata/
    │   ├── pyspark/                   integración con pandas y con PySpark, y medición
    │   └── herramientas/              página para marcar la vista de cada foto
    ├── notebooks/
    │   ├── 02-data-understanding/
    │   ├── 03-data-preparation/
    │   └── bigdata/                   comprensión, preparación, validación cruzada y medición (Colab)
    └── reports/                       fuentes en LaTeX de los reportes
        ├── 01-business-understanding/ … 04-modeling/
        ├── bigdata/
        └── privacidad-datos/
```

## CRISP-DM

| Fase | Documento | Fecha |
|---|---|---|
| 1. Business Understanding | [Versión 1](reports/01-business-understanding/business-understanding-1/) | 7 de octubre |
|  | [Versión 2](reports/01-business-understanding/business-understanding-2/Business%20Understanding.pdf) | 2 de noviembre |
|  | [Versión 3](reports/01-business-understanding/business-understanding-3/Business%20Understanding.pdf) | 12 de noviembre |
| 2. Data Understanding | [Versión 1](reports/02-data-understanding/data-understanding-1/Data%20Understanding.pdf) | 2 de noviembre |
|  | [Versión 2](reports/02-data-understanding/data-understanding-2/Data%20Understanding.pdf) | 15 de noviembre |
| 3. Data Preparation | [Versión 1](reports/03-data-preparation/data-preparation-1/Data%20Preparation.pdf) | 2 de noviembre |
| 4. Modeling | [Versión 1](reports/04-modeling/modeling-1/Modeling.pdf) | 15 de noviembre |
| 5. Evaluation | [Versión 1](reports/05-evaluation/evaluation-1/Evaluation.pdf) | 20 de noviembre |
| 6. Deployment | Pendiente |  |

Fuentes: LaTeX en [src/reports/](src/reports/), notebooks en [src/notebooks/](src/notebooks/) y código del modelado en [src/modeling/](src/modeling/).

## Módulo de Big Data

Entrega de datos del reto: la justificación del uso de herramientas de Big Data, el modelo de almacenamiento en AWS, los notebooks que extraen, limpian y cargan los datos, la validación cruzada k-fold y la medición de tiempos y recursos.

Reporte: [Justificación del uso de herramientas de Big Data](reports/bigdata/justificacion-bigdata.pdf)

| Carpeta | Contenido |
|---|---|
| [reports/bigdata/](reports/bigdata/) | Reporte y [resultados](reports/bigdata/resultados/) de los notebooks |
| [src/notebooks/bigdata/](src/notebooks/bigdata/) | Notebooks de comprensión, preparación, validación cruzada y medición (Colab) |
| [src/bigdata/pyspark/](src/bigdata/pyspark/) | Integración con pandas y con PySpark, y medición de tiempo, CPU y memoria |
| [src/bigdata/herramientas/](src/bigdata/herramientas/) | Página para marcar la vista de cada foto |
| [src/reports/bigdata/](src/reports/bigdata/) | Fuentes en LaTeX del reporte |
| [config/bigdata/](config/bigdata/) | Paquetes y configuración exportada de AWS |

### Entregable

| # | Instrucción | Dónde está |
|---|---|---|
| 1 | Herramientas y tecnologías para trabajar con los datos, y por qué | [Reporte](reports/bigdata/justificacion-bigdata.pdf) |
| 2 | Modelo de almacenamiento de los datos del reto | [Reporte](reports/bigdata/justificacion-bigdata.pdf) y la configuración exportada de AWS en [config/bigdata/aws/](config/bigdata/aws/) |
| 3 | Extraer, limpiar y cargar el conjunto de datos | [comprension_datos.ipynb](src/notebooks/bigdata/comprension_datos.ipynb), [preparacion_datos.ipynb](src/notebooks/bigdata/preparacion_datos.ipynb) (incluye la carga a S3) y [carga_resumen.json](reports/bigdata/resultados/carga_resumen.json) |
| 4 | Scripts de configuración y evidencia de que la carga funciona | [config/bigdata/](config/bigdata/), [src/bigdata/pyspark/](src/bigdata/pyspark/), [src/bigdata/herramientas/](src/bigdata/herramientas/) y la sección de carga del notebook de preparación. No hay muestra de datos, ver [Política de datos](#política-de-datos) |
| 5 | Separación en entrenamiento y prueba con k-fold | Asignación de pliegues en [preparacion_datos.ipynb](src/notebooks/bigdata/preparacion_datos.ipynb), ejecución en [validacion_cruzada.ipynb](src/notebooks/bigdata/validacion_cruzada.ipynb) y sus resultados en [reports/bigdata/resultados/](reports/bigdata/resultados/) |
| 6 | ¿Hace falta un enfoque de Big Data? | [Reporte](reports/bigdata/justificacion-bigdata.pdf), [medicion_recursos.ipynb](src/notebooks/bigdata/medicion_recursos.ipynb) y [tiempos_recursos.csv](reports/bigdata/resultados/tiempos_recursos.csv) |

Notebook individual: [ecobici_duration_pyspark.ipynb](src/notebooks/bigdata/ecobici_duration_pyspark.ipynb) (A01199004 Oswaldo Hernández).

### Política de datos

Los datos del CAETEC (fotos, bitácora de los expertos, hoja manuscrita y tablas del dataset) solo se guardan en el bucket privado de S3 del equipo, como lo define la política de datos del equipo. Por eso este repositorio no incluye los archivos de datos ni una muestra de ellos, y el nombre del bucket y las llaves de AWS no aparecen en ningún archivo.

La evidencia de que la carga funciona es la sección de carga del notebook de preparación. Esa sección compara cada archivo que genera el notebook con el que está en el bucket (por tamaño y por su huella MD5) y guarda un resumen con conteos en [carga_resumen.json](reports/bigdata/resultados/carga_resumen.json). La primera carga del dataset se hizo a mano desde la consola de S3, y la sección reproduce esa carga y la verifica.

### Flujo de los datos

1. Las fotos de las dos visitas y las hojas de BCS llegaron por carpetas de Google Drive y se subieron al bucket de S3 (región `mx-central-1`).
2. `comprension_datos.ipynb` baja los datos crudos del bucket a Colab y los describe, sin modificarlos.
3. La vista de cada foto se marcó a mano con [marcar_vistas.html](src/bigdata/herramientas/marcar_vistas.html), y el `vistas.csv` resultante se subió al bucket.
4. `preparacion_datos.ipynb` baja los datos crudos y `vistas.csv`, quita copias y exclusiones, convierte las fotos a JPG, construye la etiqueta de BCS y los pliegues, arma `dataset_bcs/` y lo compara con el del bucket.
5. `validacion_cruzada.ipynb` lee las tablas del dataset y documenta la ejecución de la validación cruzada con un modelo base.
6. `medicion_recursos.ipynb` mide tiempo, CPU y memoria de cada etapa, y compara la integración con pandas y con PySpark.

### Cómo se corre

Los notebooks corren en Google Colab. Para abrir uno desde GitHub usa Archivo, Abrir cuaderno, GitHub, y pega la liga de este repositorio. En el ícono de la llave de la barra izquierda agrega tres secretos y dales acceso al notebook.

| Secreto | Contenido |
|---|---|
| `AWS_KEY_ID` | ID de la llave de acceso de tu usuario de IAM |
| `AWS_SECRET` | Llave secreta de tu usuario de IAM |
| `S3_BUCKET` | Nombre del bucket del equipo |

El orden es comprensión, preparación, validación cruzada y medición. Los últimos tres descargan al terminar los archivos que van en [reports/bigdata/resultados/](reports/bigdata/resultados/).

Los scripts de PySpark también corren fuera de Colab, con Python 3, Java 8, 11 o 17 y los paquetes de [config/bigdata/requirements.txt](config/bigdata/requirements.txt):

```bash
cd src/bigdata/pyspark
python3 preparar_entradas.py --dataset <carpeta con los CSV del dataset> --vistas <vistas.csv> --salida entrada
python3 comparar_tiempos.py --entrada entrada --salida salida --referencia <carpeta con los CSV del dataset>
python3 comparar_tiempos.py --entrada entrada --salida salida_cluster --master spark://<ip-del-master>:7077
```

Para exportar la configuración de AWS (solo lectura y sin datos de la cuenta), quien tenga permisos sobre la cuenta corre [exportar_config_aws.py](config/bigdata/aws/exportar_config_aws.py):

```bash
python3 config/bigdata/aws/exportar_config_aws.py --bucket <nombre-del-bucket> --perfil <perfil> --salida config/bigdata/aws/exportado
```

## Módulo de Privacidad y seguridad de datos

Como parte de la actividad se encargó un reporte individual a cada integrante. Cada reporte tiene una investigación propia y una parte compartida por el equipo.

| Integrante | Reporte |
|---|---|
| Facundo Bautista Barbera | [facundo-barbera.pdf](reports/privacidad-datos/facundo-barbera.pdf) |
| Oswaldo Isaías Hernández Santes | [oswaldo-hernandez.pdf](reports/privacidad-datos/oswaldo-hernandez.pdf) |
| Alfredo Alejandro Soto Herrera | [alfredo-soto.pdf](reports/privacidad-datos/alfredo-soto.pdf) |
| Emiliano Camacho Ponce | [emiliano-camacho.pdf](reports/privacidad-datos/emiliano-camacho.pdf) |
| Iker Mejía Hernández | [iker-mejia.pdf](reports/privacidad-datos/iker-mejia.pdf) |
| Jorge Manuel Oyoqui Aguilera | [jorge-oyoqui.pdf](reports/privacidad-datos/jorge-oyoqui.pdf) |

| Carpeta | Contenido |
|---|---|
| [reports/privacidad-datos/](reports/privacidad-datos/) | Reportes individuales en PDF |
| [src/reports/privacidad-datos/compartido/](src/reports/privacidad-datos/compartido/) | Secciones compartidas por todo el equipo |
| [src/reports/privacidad-datos/alumnos/](src/reports/privacidad-datos/alumnos/) | Investigación individual de cada integrante |
| [src/reports/privacidad-datos/comun/](src/reports/privacidad-datos/comun/) | Preámbulo y portada de los reportes |

Para compilar los reportes se necesita LaTeX con `latexmk` y `biber`:

```bash
cd src/reports/privacidad-datos
make                    # compila los seis reportes
make oswaldo-hernandez  # compila solo uno
```

## Contributors

- A01066843 Facundo Bautista Barbera ([@Facundo-Barbera](https://github.com/Facundo-Barbera))
- A01199004 Oswaldo Isaías Hernández Santes ([@taqueritospro](https://github.com/taqueritospro))
- A01711368 Alfredo Alejandro Soto Herrera ([@AlejandroSH1](https://github.com/AlejandroSH1))
- A01712408 Emiliano Camacho Ponce ([@Emiliano1410](https://github.com/Emiliano1410))
- A01352358 Iker Mejía Hernández ([@ikerMJHDZ09](https://github.com/ikerMJHDZ09))
- A01711783 Jorge Manuel Oyoqui Aguilera ([@JorgeManuelOyoqui](https://github.com/JorgeManuelOyoqui))
