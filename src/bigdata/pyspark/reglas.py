"""
reglas.py - Constantes que comparten indice_pandas.py, indice_spark.py y comparar_tiempos.py.

Las columnas son las mismas que genera el notebook de preparación (notebooks/preparacion_datos.ipynb),
incluida bcs_3, así que las dos versiones de este folder y el notebook escriben tablas idénticas.
"""

K = 5  # número de pliegues de la validación cruzada

# Archivos de entrada (en --entrada) y de salida (en --salida)
ENTRADAS = ["fotos.csv", "bitacora.csv", "vistas.csv"]
SALIDAS = ["indice_imagenes.csv", "vacas.csv", "evaluaciones.csv"]

COLS_INDICE = ["ruta", "id_vaca", "vista", "usable", "usar_en_entrenamiento", "fold", "bcs_entrenamiento",
               "bcs_1", "bcs_2", "bcs_3", "n_eval", "dif_evaluadores", "archivo_jpg", "archivo_original",
               "sha256", "fecha_hora", "camara", "ancho", "alto", "profundidad", "marcado_por", "marcado_en"]
COLS_VACAS = ["id_vaca", "en_entrenamiento", "fold", "bcs_entrenamiento", "bcs_1", "bcs_2", "bcs_3", "n_eval",
              "dif_evaluadores", "n_fotos", "n_usables", "n_posterior", "n_lateral_izq", "n_lateral_der"]
COLS_EVAL = ["fila_excel", "id_vaca", "hora", "bcs", "orden_evaluacion", "n_eval", "fold", "en_entrenamiento"]

# Columnas que se comparan como número, para que "3.0" y "3" cuenten igual
NUMERICAS = {"fold", "bcs_entrenamiento", "bcs_1", "bcs_2", "bcs_3", "n_eval", "dif_evaluadores", "ancho", "alto",
             "n_fotos", "n_usables", "n_posterior", "n_lateral_izq", "n_lateral_der", "bcs", "orden_evaluacion",
             "fila_excel"}

# Columnas con las que se ordenan las filas antes de comparar dos versiones de una tabla
LLAVES = {"indice_imagenes.csv": ["ruta"],
          "vacas.csv": ["id_vaca"],
          "evaluaciones.csv": ["id_vaca", "orden_evaluacion"]}

MAX_EVALUACIONES = 3  # bcs_1 a bcs_3; si alguna vaca tiene más, hay que agregar otra columna
