# Practica 2 - Repo 1: Modelado, Calibracion e Incertidumbre

Pipeline de modelado para **deteccion de impago de prestamos**. Parte de los artefactos
de preprocesamiento y filtrado de la Practica 1 y produce el modelo final que sirve
la API del Repo 2.

## Que hace

El notebook `practica2_notebook.ipynb` ejecuta de principio a fin:

1. **Optimizacion con Optuna** (variacion del notebook de clase `11_optuna_tpe.ipynb`):
   optimiza **Log Loss** con TPE + `HyperbandPruner`, espacio de busqueda ampliado, y
   compara cada modelo **con y sin** tratamiento de desbalanceo. Modelos: LightGBM y XGBoost.
2. **Calibracion** (variacion de `10_calibracion.ipynb`): diagnostico (reliability diagram,
   ECE, Brier, test de Spiegelhalter) y decision argumentada de calibrar o no.
3. **Incertidumbre** (variacion de `09_conformal_prediction.ipynb`): intervalo
   `[p_low, p_high]` por cliente con **Venn-Abers**, y politica de derivacion a un agente
   cuando la anchura supera 0.2.
4. **Persistencia**: guarda `artifacts/practica2_model.pkl` y `artifacts/feature_schema.json`.

## Estructura del repo

```
practica2-modelado/
├── practica2_notebook.ipynb     # el notebook (ejecutar de principio a fin)
├── pyproject.toml               # dependencias (gestionadas con uv)
├── README.md
├── src/
│   ├── preprocessing/
│   │   └── base_preprocessing.py   # clase BasePreprocess (Practica 1)
│   ├── filtering/
│   │   └── base_filtering.py       # clase BaseFiltering (Practica 1)
│   └── practica2_model.py          # clase Practica2Model (artefacto final)
├── data/                        # <-- colocar aqui los datos (ver abajo)
└── artifacts/                   # <-- el notebook escribe aqui los .pkl/.json
```

## Archivos necesarios (no incluidos en el repo)

Coloca en la **raiz del repo** estos ficheros, que vienen del modulo de la practica:

| Fichero | Que es |
|---|---|
| `df_train_small.csv` | datos de entrenamiento crudos |
| `df_test_small.csv` | datos de test crudos |
| `preprocessor.pkl` | objeto `BasePreprocess` ya fitteado (Practica 1) |
| `filter.pkl` | objeto `BaseFiltering` ya fitteado (Practica 1) |
| `variables_withExperts.xlsx` | catalogo de variables (lo usa el preprocessor internamente) |

> Los `.pkl` se cargan, **no se re-fittean**: el notebook solo aplica `.transform()`.

## Instalacion y ejecucion (local, con uv)

```bash
# 1. Instalar uv si no lo tienes: https://docs.astral.sh/uv/
# 2. Crear el entorno e instalar dependencias
uv sync

# 3. Lanzar Jupyter
uv run jupyter notebook practica2_notebook.ipynb
```

Ejecuta las celdas en orden (`Kernel > Restart & Run All`). Al terminar, en `artifacts/`
tendras:

- `practica2_model.pkl` — el modelo final (con `predict`, `predict_proba`, `predict_interval`).
- `feature_schema.json` — esquema de columnas crudas + metadatos, que consume la API.

## Notas importantes

- **Primera ejecucion con internet.** El `TextEncoder` de `skrub` (usado dentro de
  `preprocessor.pkl`) descarga el modelo `intfloat/e5-small-v2` de HuggingFace la primera
  vez. Despues queda en cache local.
- **Versiones de librerias.** Si `joblib.load("preprocessor.pkl")` falla, suele ser por
  desajuste de version de `scikit-learn` / `skrub` / `feature-engine` respecto a las que
  generaron el pickle. Ajusta las versiones en `pyproject.toml` a las del repo de Practica 1.
- **Reproducibilidad.** Todos los `random_state` / `seed` estan fijados a 42. Las
  dependencias quedan congeladas en `uv.lock` tras `uv sync`.
- **Tiempo de ejecucion.** Son 4 studies de Optuna de 30 trials cada uno. En una maquina
  normal, del orden de minutos. Para acelerar, baja `N_TRIALS` en la celda correspondiente.
