# app/core/model_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set
import re
import ast

import numpy as np
import pandas as pd

# Algunas rutas importan a feature_map. Si no existe (o aún no compila), hacemos fallbacks.
try:
    from .feature_map import MODEL_FEATURES  # columnas finales para sklearn
except Exception:
    MODEL_FEATURES = []

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _get_numeric_cols_fallback() -> Set[str]:
    """
    Conjunto de columnas que muy probablemente fueron tratadas como numéricas
    durante el entrenamiento con PyCaret (fallback si no podemos importar).
    """
    return {
        "presupuesto_estimado", "frecuencia_viaje", "sitios_visitados",
        "calificacion_sitios_previos", "tiempo_estancia_promedio",
        "afluencia_promedio", "ratio_costo_presu",
        "costo_entrada", "duracion_esperada", "edad",
        "tiempo_disponible_min", "match_mascotas"
    }

def _list_train_numeric_cols() -> Set[str]:
    """
    Trae TRAIN_NUMERIC_COLS desde feature_map si es posible, si no, usa fallback.
    """
    try:
        from .feature_map import TRAIN_NUMERIC_COLS
        return set(TRAIN_NUMERIC_COLS)
    except Exception:
        return _get_numeric_cols_fallback()

def _add_missing_cols_for_pycaret(X_: pd.DataFrame, keyerror_msg: str) -> pd.DataFrame:
    """
    Cuando PyCaret lanza KeyError pidiendo columnas, este helper las crea.
    Numéricas -> np.nan; Categóricas -> '' (vacío). Luego dejamos que el pipeline
    impute / transforme.
    """
    m = re.search(r"\[(.*?)\]", keyerror_msg)
    if not m:
        return X_

    raw = "[" + m.group(1) + "]"
    try:
        cols = ast.literal_eval(raw)  # ['a','b', ...]
    except Exception:
        cols = [c.strip().strip("'\"") for c in m.group(1).split(",")]

    num_set = _list_train_numeric_cols()
    X_ = X_.copy()
    for c in cols:
        if c not in X_.columns:
            X_[c] = (np.nan if c in num_set else "")
    return X_

def _sanitize_dataframe_for_pycaret(X: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza GLOBAL para evitar errores comunes en pipelines de PyCaret/sklearn:
    - '' y espacios -> np.nan
    - pd.NA -> np.nan
    - columnas con StringDtype -> object
    - fuerza numéricas conocidas a float (to_numeric)
    - autodetección: columnas object que parecen numéricas -> to_numeric
    """
    X_ = X.copy()

    # 0) Vacíos y NAType
    X_ = X_.replace(to_replace=r"^\s*$", value=np.nan, regex=True).replace({pd.NA: np.nan})

    # 1) Columnas 'string' -> 'object' (evita NAType en __array__)
    for c in X_.columns:
        if pd.api.types.is_string_dtype(X_[c].dtype):
            X_[c] = X_[c].astype("object")

    # 2) Fuerza numéricas conocidas
    num_cols = _list_train_numeric_cols()
    for c in num_cols:
        if c in X_.columns:
            X_[c] = pd.to_numeric(X_[c], errors="coerce")

    # 3) Autodetección: object que parecen numéricas
    obj_cols = [c for c in X_.columns if X_[c].dtype == "object"]
    for c in obj_cols:
        s = X_[c].dropna()
        if len(s) == 0:
            continue
        s_stripped = s.astype(str).str.strip()
        s_stripped = s_stripped[s_stripped != ""]
        if len(s_stripped) and pd.to_numeric(s_stripped, errors="coerce").notna().all():
            X_[c] = pd.to_numeric(X_[c], errors="coerce")

    return X_

def _extract_score_column(preds: pd.DataFrame) -> pd.Series:
    """
    PyCaret 2.x devuelve 'Score'; PyCaret 3.x suele devolver 'prediction_score'.
    Además, algunos modelos exponen 'probability', 'probability_1', etc.
    Este helper selecciona la mejor columna disponible y devuelve una Serie float.
    """
    # candidatos típicos
    for cand in ["Score", "prediction_score", "probability", "probability_1", "Score_1"]:
        if cand in preds.columns:
            return preds[cand].astype(float)

    # Fallback: si solo hay etiqueta, úsala como 0/1
    for lab in ["Label", "prediction_label"]:
        if lab in preds.columns:
            return preds[lab].astype(int).astype(float)

    # Último recurso: cualquier columna numérica con 'score'/'prob' en el nombre
    prob_like = [
        c for c in preds.columns
        if (("score" in c.lower() or "prob" in c.lower()) and preds[c].dtype.kind in "fc")
    ]
    if prob_like:
        return preds[prob_like[0]].astype(float)

    raise RuntimeError(f"No encontré columna de score en predict_model. Columnas: {list(preds.columns)}")


# --------------------------------------------------------------------------------------
# UniversalModel
# --------------------------------------------------------------------------------------

class UniversalModel:
    """
    Carga un modelo entrenado con:
    - PyCaret (preferred): load_model(base_path)  -> self.kind = 'pycaret'
    - scikit-learn (joblib/pickle): joblib.load(path) -> self.kind = 'sklearn'

    Predicción:
    - PyCaret: usa predict_model con sanitización y resolución flexible del 'score'.
    - Sklearn: usa predict_proba si existe, si no, predict (y lo castea a float).
    """

    def __init__(self, path: str | Path):
        self.kind: Optional[str] = None
        self.model = None

        p = Path(path)
        self._no_ext = str(p.with_suffix(""))  # PyCaret suele guardar sin depender de la extensión

        # 1) Intento PyCaret
        try:
            from pycaret.classification import load_model
            self.model = load_model(self._no_ext)
            self.kind = "pycaret"
            print("Transformation Pipeline and Model Successfully Loaded")
            return
        except Exception:
            self.model = None

        # 2) Intento scikit-learn (joblib)
        try:
            import joblib
            if p.suffix.lower() == ".pkl" and p.exists():
                self.model = joblib.load(str(p))
            else:
                self.model = joblib.load(self._no_ext + ".pkl")
            self.kind = "sklearn"
            print("Sklearn Model Successfully Loaded")
            return
        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el modelo desde '{p}': {e}")

    # ------------------------------------------------------------------

    def predict_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Devuelve puntajes de preferencia (probabilidad de clase positiva).
        - Para PyCaret: sanitiza, reintenta si faltan columnas y resuelve el nombre del score.
        - Para Sklearn: usa predict_proba si está disponible.
        """
        X_ = X.copy()

        # --- PyCaret ---
        if self.kind == "pycaret":
            from pycaret.classification import predict_model

            # Sanitización global agresiva (''/pd.NA -> np.nan, coerción numérica, etc.)
            X_ = _sanitize_dataframe_for_pycaret(X_)

            # Predict con retry si faltan columnas
            try:
                # raw_score=False -> prob estándar
                preds = predict_model(self.model, data=X_, raw_score=False)
            except KeyError as ke:
                # crea columnas faltantes (num -> NaN, cat -> '')
                X_ = _add_missing_cols_for_pycaret(X_, str(ke))
                # re-sanitiza y reintenta
                X_ = _sanitize_dataframe_for_pycaret(X_)
                preds = predict_model(self.model, data=X_, raw_score=False)

            # Seleccionar la columna de score que corresponda
            return _extract_score_column(preds)

        # --- Sklearn ---
        if hasattr(self.model, "predict_proba"):
            # Si no hay MODEL_FEATURES, usamos todas las columnas
            feats = MODEL_FEATURES if MODEL_FEATURES else list(X_.columns)
            proba = self.model.predict_proba(X_[feats])[:, 1]
            return pd.Series(proba, index=X_.index, name="score")

        # Fallback a predict() (0/1) y lo devolvemos como float
        feats = MODEL_FEATURES if MODEL_FEATURES else list(X_.columns)
        yhat = self.model.predict(X_[feats])
        return pd.Series(yhat, index=X_.index, name="score").astype(float)
