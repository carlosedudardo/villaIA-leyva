
# -*- coding: utf-8 -*-
"""
Evaluación del modelo de clasificación (y_like) para recomendaciones en Villa de Leyva.

Qué hace este script:
1) Carga el dataset central y replica el *mismo* feature engineering del notebook.
2) Construye un hold-out honesto por usuario (id_usuario) y métricas Top-K por usuario.
3) Intenta cargar tu modelo de PyCaret guardado ("modelo_cls_like_v2").
   - Si existe, evalúa con predict_model (PyCaret).
   - Si NO existe, puede entrenar un baseline scikit-learn (opcional) para comparar.
4) Imprime métricas de clasificación (Accuracy, Precision, Recall, F1, ROC AUC) y de ranking (Recall@K, NDCG@K, Coverage@K).
5) Guarda artefactos en la carpeta ./eval_out:
   - metrics_clf.json, metrics_topk.json
   - confusion_matrix.png
   - predicciones_holdout.csv  (con probabilidades por usuario-sitio)

Cómo ejecutarlo (en tu entorno Anaconda con PyCaret):
    conda activate <tu_env>
    pip install pycaret "pandas<2.2" numpy scikit-learn matplotlib
    python eval_model_vdl.py --data "dataset_Recomendacion_villa_de_leyva_eleccion (2).csv" --model_name "modelo_cls_like_v2" --sep ";" --enc "utf-8-sig"
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------
# Config por defecto
# ---------------------
DEFAULT_DATA = "dataset_Recomendacion_villa_de_leyva_eleccion (2).csv"
DEFAULT_SEP  = ";"
DEFAULT_ENC  = "utf-8-sig"
DEFAULT_MODEL_NAME = "modelo_cls_like_v2"   # lo que usaste en save_model(...)

K_LIST = [3, 5, 10]
TEST_USER_FRAC = 0.20
RANDOM_SEED = 42

# ---------------------
# Utilidades Notebook
# ---------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {
        "compa¤¡a_viaje": "compania_viaje",
        "‚poca_visita": "epoca_visita",
    }
    ren = {k: v for k, v in ren.items() if k in df.columns}
    return df.rename(columns=ren)

CAT_COLS = [
    "nacionalidad","origen","tipo_turista_preferido","compania_viaje",
    "restricciones_movilidad","nombre_sitio","tipo_sitio","accesibilidad_general",
    "idioma_info","ubicacion_geografica","clima_predominante","epoca_visita"
]
NUM_COLS = [
    "edad","frecuencia_viaje","presupuesto_estimado","sitios_visitados",
    "calificacion_sitios_previos","tiempo_estancia_promedio","costo_entrada",
    "afluencia_promedio","duracion_esperada","admite_mascotas"
]

AFINIDAD = {
    "cultural": {"museo":0.9,"centro_historico":0.9,"arquitectura":0.85,"arqueologico":0.85,"plaza":0.7,"religioso":0.7},
    "naturaleza": {"naturaleza":0.95,"senderismo":0.9,"mirador":0.8},
    "aventura": {"senderismo":0.9,"parque_tematico":0.75,"mirador":0.75,"naturaleza":0.7},
    "gastronomico": {"gastronomico":0.95},
    "relax": {"mirador":0.9,"plaza":0.8,"naturaleza":0.75,"arquitectura":0.8},
}

def make_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    denom = (X["presupuesto_estimado"]*0.15).replace(0, np.nan)
    X["ratio_costo_presu"] = (X["costo_entrada"] / denom).clip(0, 3).fillna(0)

    X["afinidad_tipo"] = X.apply(
        lambda r: AFINIDAD.get(str(r["tipo_turista_preferido"]), {}).get(str(r["tipo_sitio"]), 0.5), axis=1
    )
    X["x_tipoTur__tipoSit"] = X["tipo_turista_preferido"].astype(str) + "×" + X["tipo_sitio"].astype(str)
    X["x_epoca__tipoSit"]   = X["epoca_visita"].astype(str) + "×" + X["tipo_sitio"].astype(str)
    return X

def _dcg_at_k(rels): 
    return float(np.sum([r/np.log2(i+2) for i, r in enumerate(rels)]))

def recall_at_k(g, k, score_col, rel_col):
    g = g.sort_values(score_col, ascending=False)
    topk = g.head(k)
    tot = g[rel_col].sum()
    return float("nan") if tot == 0 else float(topk[rel_col].sum()/tot)

def ndcg_at_k(g, k, score_col, rel_col):
    g = g.sort_values(score_col, ascending=False)
    dcg  = _dcg_at_k(g.head(k)[rel_col].tolist())
    idcg = _dcg_at_k(sorted(g[rel_col].tolist(), reverse=True)[:k])
    return float("nan") if idcg == 0 else float(dcg/idcg)

def coverage_at_k(df, k, score_col, item_col="nombre_sitio"):
    topk = (df.sort_values(["id_usuario", score_col], ascending=[True, False])
              .groupby("id_usuario").head(k))
    return float(topk[item_col].nunique() / df[item_col].nunique())

def split_users_holdout(df: pd.DataFrame, test_frac: float, random_state: int = 42):
    users = df["id_usuario"].drop_duplicates().sample(frac=1.0, random_state=random_state)
    n_test = max(1, int(len(users) * test_frac))
    test_users = set(users.iloc[:n_test].astype(str))
    mask = df["id_usuario"].astype(str).isin(test_users)
    return df.loc[~mask].copy(), df.loc[mask].copy()

def ensure_types(df: pd.DataFrame, cat_cols, num_cols):
    df = df.copy()
    for c in cat_cols:
        if c in df.columns: df[c] = df[c].astype("string")
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return df

def eval_topk(per_user_df, score_col="Score", rel_col="y_like", k_list=(3,5,10)):
    # per_user_df: (id_usuario, nombre_sitio, Score, y_like)
    metrics = {}
    gb = per_user_df.groupby("id_usuario", group_keys=False)
    for k in k_list:
        rec = gb.apply(lambda g: recall_at_k(g, k, score_col, rel_col)).mean()
        ndc = gb.apply(lambda g: ndcg_at_k(g, k, score_col, rel_col)).mean()
        cov = coverage_at_k(per_user_df, k, score_col, item_col="nombre_sitio")
        metrics[f"recall@{k}"]  = float(rec)
        metrics[f"ndcg@{k}"]    = float(ndc)
        metrics[f"coverage@{k}"]= float(cov)
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--sep", default=DEFAULT_SEP)
    ap.add_argument("--enc", default=DEFAULT_ENC)
    ap.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--no_baseline", action="store_true", help="No entrenar baseline si no hay modelo PyCaret")
    args = ap.parse_args()

    out_dir = Path("eval_out"); out_dir.mkdir(exist_ok=True)

    # --- Carga datos y FE ---
    df = pd.read_csv(args.data, sep=args.sep, encoding=args.enc)
    df = normalize_columns(df)
    df = make_features(df)

    # etiqueta y_like desde rating_usuario
    df["y_like"] = (df["rating_usuario"] >= 4.0).astype(int)

    # tipos
    EXT_CAT = CAT_COLS + ["x_tipoTur__tipoSit","x_epoca__tipoSit"]
    EXT_NUM = NUM_COLS + ["ratio_costo_presu","afinidad_tipo"]

    df = ensure_types(df, EXT_CAT, EXT_NUM)

    # split por usuario
    train_df, test_df = split_users_holdout(df, TEST_USER_FRAC, RANDOM_SEED)

    # --- Intento: cargar modelo PyCaret ---
    has_pycaret = False
    preds = None
    try:
        from pycaret.classification import load_model, predict_model
        has_pycaret = True
    except Exception as e:
        print("[AVISO] PyCaret no disponible en este entorno:", e)

    model_loaded = None
    if has_pycaret:
        try:
            model_loaded = load_model(args.model_name)
            print(f"[OK] Modelo PyCaret cargado: {args.model_name}")
        except Exception as e:
            print(f"[AVISO] No encontré el modelo '{args.model_name}'. Error: {e}")

    # --- Evaluación ---
    if model_loaded is not None:
        # predict_model espera columnas tal como en el setup original
        feat_cols = EXT_CAT + EXT_NUM
        Xtest = test_df[["id_usuario","nombre_sitio"] + feat_cols + ["y_like"]].copy()
        out  = predict_model(model_loaded, data=Xtest, raw_score=True)  # incluye 'Score' (prob clase positiva)
        # Asegurar columnas para métricas Top-K
        per_user = out[["id_usuario","nombre_sitio","y_like","Score"]].copy()
        score_col = "Score"
        y_true = out["y_like"].astype(int).to_numpy()
        y_prob = out[score_col].astype(float).to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)

    else:
        if args.no_baseline:
            raise SystemExit("No hay modelo de PyCaret y --no_baseline está activo. Nada que evaluar.")
        print("[BASELINE] Entrenando LogisticRegression (scikit-learn) para referencia...")
        # One-hot de categóricas con pipeline
        feat_cols = EXT_CAT + EXT_NUM
        X = train_df[feat_cols].copy()
        y = (train_df["y_like"]).astype(int).to_numpy()
        pre = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), EXT_CAT)],
            remainder="passthrough"
        )
        clf = LogisticRegression(max_iter=200, n_jobs=1)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        # GroupKFold para no mezclar usuarios en CV (5 folds)
        gkf = GroupKFold(n_splits=5)
        groups = train_df["id_usuario"].astype(str).to_numpy()
        # Entrenar (simple, sin CV para ahorrar tiempo aquí)
        pipe.fit(X, y)
        # Predicciones hold-out
        Xtest = test_df[["id_usuario","nombre_sitio"] + feat_cols + ["y_like"]].copy()
        y_true = Xtest["y_like"].astype(int).to_numpy()
        y_prob = pipe.predict_proba(Xtest[feat_cols])[:,1]
        y_pred = (y_prob >= 0.5).astype(int)
        per_user = Xtest[["id_usuario","nombre_sitio","y_like"]].copy()
        per_user["Score"] = y_prob
        score_col = "Score"

    # --- Métricas de clasificación ---
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_fscore_support
    metrics_clf = {}
    metrics_clf["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    metrics_clf["accuracy"] = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics_clf["precision"] = float(prec)
    metrics_clf["recall"] = float(rec)
    metrics_clf["f1"] = float(f1)

    print("\n=== Métricas de clasificación (hold-out por usuario) ===")
    for k,v in metrics_clf.items():
        print(f"{k:>10}: {v:.4f}")

    # --- Reporte y matriz de confusión ---
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión (hold-out)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # --- Métricas Top-K por usuario ---
    metrics_topk = eval_topk(per_user, score_col=score_col, rel_col="y_like", k_list=K_LIST)
    print("\n=== Métricas de ranking (por usuario) ===")
    for k,v in metrics_topk.items():
        print(f"{k:>10}: {v:.4f}")

    # Guardar artefactos
    (out_dir / "metrics_clf.json").write_text(json.dumps(metrics_clf, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics_topk.json").write_text(json.dumps(metrics_topk, indent=2, ensure_ascii=False), encoding="utf-8")

    # Guardar predicciones por usuario-sitio (para auditoría)
    per_user.to_csv(out_dir / "predicciones_holdout.csv", index=False, encoding="utf-8-sig")

    print(f"\nListo. Archivos guardados en: {out_dir.resolve()}")
    if model_loaded is None:
        print("\nNOTA: Estos resultados corresponden al baseline scikit-learn.")
        print("      Para evaluar *tu* modelo de PyCaret, vuelve a correr el script")
        print("      asegurando que 'modelo_cls_like_v2' exista en esta misma carpeta.")

if __name__ == "__main__":
    main()
