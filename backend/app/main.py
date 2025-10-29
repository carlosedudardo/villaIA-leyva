# app/main.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import pandas as pd

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .core.model_loader import UniversalModel
from .core.recommender import RecommenderService

# ------------------ Paths / Config ------------------
# BASE_DIR -> /Backend
BASE_DIR = Path(__file__).resolve().parent.parent
# APP_DIR  -> /Backend/app
APP_DIR = Path(__file__).resolve().parent
# DATA_DIR -> /Backend/app/data
DATA_DIR = APP_DIR / "data"

MODEL_PATH = BASE_DIR / "models" / "modelo_cls_like_v3.pkl"  # ajusta si cambia el nombre

# Permite sobreescribir el catálogo por variable de entorno (ruta absoluta o relativa)
DEFAULT_CATALOG_NAME = "catalogo_vdl_lugares_unico.csv"
CAT_PATH = Path(os.getenv("VDL_CATALOG", DATA_DIR / DEFAULT_CATALOG_NAME))

def _load_catalog(path: Path) -> pd.DataFrame:
    """Carga el catálogo intentando primero con ';' y luego con ','."""
    if not path.exists():
        # fallback: cualquier .csv en app/data
        candidates = list(DATA_DIR.glob("*.csv"))
        if candidates:
            print(f"[INFO] CAT_PATH no encontrado, usando {candidates[0]}")
            path = candidates[0]
        else:
            raise FileNotFoundError(
                f"No se encontró catálogo en {path}. "
                f"Coloca un CSV en {DATA_DIR} o exporta VDL_CATALOG con la ruta."
            )
    try:
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", engine="python")
        return df
    except Exception:
        # intenta con coma
        df = pd.read_csv(path, sep=",", encoding="utf-8-sig", engine="python")
        return df

# ------------------ App ---------------------
app = FastAPI(title="Villa de Leyva Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción especifica tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Carga recursos ----------

_catalog = _load_catalog(CAT_PATH)
print(f"[OK] Catálogo cargado desde: {CAT_PATH}  (filas={len(_catalog)})")

_model = UniversalModel(MODEL_PATH)
_reco = RecommenderService(_model, _catalog)
print("[OK] Modelo cargado y servicio inicializado")

# ------------------ Schemas -----------------
class SurveyIn(BaseModel):
    edad: Optional[int] = None
    nacionalidad: Optional[str] = "Colombia"
    origen: Optional[str] = None
    tipo_turista_preferido: Optional[str] = None
    compania_viaje: Optional[str] = None
    epoca_visita: Optional[str] = None
    presupuesto_estimado: Optional[float] = None
    restricciones_movilidad: Optional[str] = None
    tiempo_disponible_min: Optional[int] = 120
    admite_mascotas: Optional[bool] = None
    # agrega aquí cualquier otro campo que envíe tu UI

# ------------------ Rutas -------------------
@app.get("/")
def root():
    return {"ok": True, "service": "Villa de Leyva Recommender API"}

@app.post("/recommend")
def recommend(payload: SurveyIn, top_n: int = Query(5, ge=1, le=20)):
    # 1) Convierte el body en dict (excluye None para no ensuciar)
    survey = payload.model_dump(exclude_none=True)

    # 2) Ejecuta el servicio de recomendación
    recs_df = _reco.recommend(survey, top_n)

    # 3) Devuelve JSON (lista de objetos)
    return JSONResponse(content=recs_df.to_dict(orient="records"))
