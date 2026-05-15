"""
practica2_model.py
==================
Clase `Practica2Model`: el artefacto final de la Practica 2.

Envuelve en un solo objeto serializable:
  1. El modelo base ganador (LightGBM o XGBoost, ya entrenado).
  2. Un calibrador post-hoc opcional (Platt / Isotonic), si en la Seccion 2 se
     decidio calibrar.
  3. Una capa Venn-Abers que produce el intervalo [p_low, p_high] por prediccion.

Metodos que la API del Repo 2 consume:
  - predict_proba(X)    -> ndarray (n, 2)
  - predict(X)          -> ndarray (n,) con 0/1
  - predict_interval(X) -> (p_low, p_high)

Este fichero se incluye TANTO en el Repo 1 como en el Repo 2, porque joblib
necesita la definicion de la clase para deserializar el .pkl.
"""
from __future__ import annotations
import numpy as np
from venn_abers import VennAbers


class Practica2Model:
    def __init__(self, base_model, calibrator=None):
        """
        Parametros
        ----------
        base_model : estimador sklearn-like YA entrenado (LGBM / XGB).
        calibrator : CalibratedClassifierCV YA entrenado o None.
                     Si se pasa, predict_proba usa el calibrador.
        """
        self.base_model = base_model
        self.calibrator = calibrator
        self._va = None    # se ajusta en fit_interval()

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        """ndarray (n, 2): [P(no impago), P(impago)]."""
        estimator = self.calibrator if self.calibrator is not None else self.base_model
        return np.asarray(estimator.predict_proba(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    # ------------------------------------------------------------------
    def fit_interval(self, X_cal, y_cal):
        """Ajusta Venn-Abers sobre el conjunto de calibracion."""
        y_cal = np.asarray(y_cal).astype(int)
        p_cal_2col = np.asarray(self.base_model.predict_proba(X_cal))
        self._va = VennAbers()
        self._va.fit(p_cal_2col, y_cal)
        return self

    def predict_interval(self, X):
        """Devuelve (p_low, p_high) por fila."""
        if self._va is None:
            raise RuntimeError("Llama a fit_interval(X_cal, y_cal) antes de predict_interval().")
        p_test_2col = np.asarray(self.base_model.predict_proba(X))
        _, p0p1 = self._va.predict_proba(p_test_2col)
        p0p1 = np.asarray(p0p1)
        return p0p1[:, 0], p0p1[:, 1]

    def __repr__(self):
        return (f"Practica2Model(base={type(self.base_model).__name__}, "
                f"calibrator={type(self.calibrator).__name__ if self.calibrator else None})")
