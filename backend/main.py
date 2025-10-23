from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models_schemas import PerfilUsuario
from recommender import RecommenderService
import numpy as np


app = FastAPI(title="API Recomendador VDL")

# CORS para que puedas consumir desde React/Angular/Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

service = RecommenderService("modelo_cls_like_v3.pkl", "catalogo_vdl_lugares_unico.csv")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/recommend")
def recommend(user: PerfilUsuario):
    top = service.topn(user)
    return {
        "items": top.to_dict(orient="records"),
        "total_catalogo": int(service.catalog.shape[0])
    }

@app.post("/_debug_preview")
def debug_preview(user: PerfilUsuario):
    import numpy as np  # <- puedes dejarlo aquí o arriba como global
    X = service._rows_from_user(user)
    nunique = X.nunique().to_dict()
    stats = {
        "shape": list(X.shape),
        "nunique_by_col": nunique,
        "head": X.head(3).to_dict(orient="list"),
    }
    try:
        if hasattr(service.model, "predict_proba"):
            p = service.model.predict_proba(X)[:, -1]
        else:
            p = service.model.predict(X)

        p = np.array(p, dtype=float)
        stats["pred_stats"] = {
            "min": float(p.min()),
            "max": float(p.max()),
            "std": float(p.std()),
            "unique": int(np.unique(p).size),
            "sample": p[:5].tolist()
        }
    except Exception as e:
        stats["pred_error"] = str(e)
    return stats
