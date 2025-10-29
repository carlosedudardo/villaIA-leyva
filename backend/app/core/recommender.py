# app/core/recommender.py
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from .feature_map import (
    build_pair_matrix,
    ensure_training_columns,
    ensure_features,
    MODEL_FEATURES,
)

DISPLAY_COLS: List[str] = [
    "nombre_sitio",
    "tipo_sitio",
    "ubicacion_geografica",
    "costo_entrada",
    "admite_mascotas",
    "idioma_info",
    "score_like",
]

class RecommenderService:
    """
    Orquesta el armado del par (usuario, sitio) a partir del catálogo,
    garantiza columnas para el pipeline (PyCaret/sklearn), predice y devuelve
    un DataFrame con las columnas exactas que el frontend espera.
    """

    def __init__(self, model: Any, catalog_df: pd.DataFrame):
        self.model = model
        # guardamos una copia “clean” del catálogo
        self.catalog = catalog_df.copy()

    # ------------------------------- públicos ------------------------------- #

    def recommend(self, survey_ui: Dict, top_n: int) -> pd.DataFrame:
        """
        1) Construye matriz (catalog + UI)
        2) Garantiza columnas de entrenamiento para PyCaret
        3) Garantiza tipos/orden de features del estimador
        4) Predice con el modelo universal (PyCaret/sklearn)
        5) Retorna Top-N con esquema estable para el frontend
        """
        # 1) Matriz completa
        X_raw = build_pair_matrix(self.catalog, survey_ui)

        # 2) Columnas de entrenamiento (las que PyCaret recuerda)
        X_raw = ensure_training_columns(X_raw)

        # 3) Features finales del estimador
        X_feats = ensure_features(X_raw)

        # 4) Inyecta las MODEL_FEATURES tipadas dentro de X_raw (PyCaret las mirará)
        for c in MODEL_FEATURES:
            X_raw[c] = X_feats[c]

        # 5) Predice
        scores = self.model.predict_scores(X_raw)

        # 6) Ensambla salida visible y ordena
        out = self._build_display_frame(X_raw, scores)
        out = out.sort_values("score_like", ascending=False).head(int(top_n))

        # 7) Devuelve exactamente las columnas que el front consume
        return out[DISPLAY_COLS].reset_index(drop=True)

    # ------------------------------ helpers -------------------------------- #

    def _build_display_frame(self, X_raw: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
        """
        Arma el DataFrame de salida, creando columnas faltantes con defaults
        y normalizando tipos para que el JSON sea estable.
        """
        df = X_raw.copy()
        df["score_like"] = scores.values

        # Asegura TODAS las columnas visibles
        for c in DISPLAY_COLS:
            if c not in df.columns:
                # defaults razonables
                if c == "score_like":
                    df[c] = 0.0
                elif c in ("costo_entrada",):
                    df[c] = 0.0
                elif c in ("admite_mascotas",):
                    df[c] = 0
                else:
                    df[c] = ""

        # Normaliza tipos para serializar bien
        df["costo_entrada"] = pd.to_numeric(df["costo_entrada"], errors="coerce").fillna(0.0)
        # admite_mascotas -> 0/1
        df["admite_mascotas"] = (
            df["admite_mascotas"]
            .apply(lambda x: 1 if str(x).strip().lower() in ("1", "true", "sí", "si") else 0)
            .astype(int)
        )

        # Fallbacks para campos de texto (evitar nulos/NA)
        for c in ["nombre_sitio", "tipo_sitio", "ubicacion_geografica", "idioma_info"]:
            df[c] = df[c].astype(object).where(df[c].notna(), "")
            df[c] = df[c].replace({np.nan: "", None: ""})

        # Si el catálogo no tenía nombre, inventa uno mínimo para no dejar la tarjeta en blanco
        # (opcional; puedes quitar si prefieres dejar "Nombre no disponible" en el front)
        mask_empty = df["nombre_sitio"].astype(str).str.strip().eq("")
        if mask_empty.any():
            df.loc[mask_empty, "nombre_sitio"] = (
                df.loc[mask_empty, ["tipo_sitio", "ubicacion_geografica"]]
                .apply(lambda r: f"{(r['tipo_sitio'] or 'sitio').strip()} - {(r['ubicacion_geografica'] or '').strip()}".strip(" -"),
                       axis=1)
            )

        # Asegura float en score_like
        df["score_like"] = pd.to_numeric(df["score_like"], errors="coerce").fillna(0.0)

        return df
