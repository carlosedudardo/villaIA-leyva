import pandas as pd
import numpy as np
import joblib

_PRESUP_MAP = {"bajo": 20000, "medio": 50000, "alto": 100000}

def _read_catalog_robusto(path: str) -> pd.DataFrame:
    seps = [None, ";", "\t", ","]
    for s in seps:
        try:
            df = pd.read_csv(path, sep=s, engine="python", encoding="utf-8-sig")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8-sig")

def _afinidad_por_tipo(interes_ppal: str, tipo_sitio: str) -> float:
    interes_ppal = (interes_ppal or "").lower()
    tipo = (tipo_sitio or "").lower()

    grupos = {
        "naturaleza": {"natural","sendero","parque","pozos","cascada"},
        "historia": {"museo","iglesia","monumento","histórico","fósil","arquitectura"},
        "gastronomía": {"gastronomia","restaurante","mercado","café"},
        "arte": {"arte","arquitectura","galeria","terracota"},
        "aventura": {"aventura","deporte","extremo","caverna","bike"},
    }
    h = 1.0 if any(v in tipo for v in grupos.get(interes_ppal, set())) else 0.2
    return h


class RecommenderService:
    def __init__(self, model_path="modelo_cls_like_v3.pkl", cat_path="catalogo_vdl_lugares_unico.csv"):
        self.model = joblib.load(model_path)

        cat = _read_catalog_robusto(cat_path)
        cat.columns = cat.columns.str.strip()
        print("CATALOGO COLS:", list(cat.columns))

        # Normaliza mínimos que vienen en tu CSV
        req = ["nombre_sitio","tipo_sitio","costo_entrada","afluencia_promedio","duracion_esperada","admite_mascotas",
               "accesibilidad_general","idioma_info","ubicacion_geografica","clima_predominante"]
        for c in req:
            if c not in cat.columns:
                cat[c] = "desconocido" if c in ["nombre_sitio","tipo_sitio","accesibilidad_general","idioma_info","ubicacion_geografica","clima_predominante"] else 0

        num_cols = ["costo_entrada","afluencia_promedio","duracion_esperada"]
        for c in num_cols:
            cat[c] = pd.to_numeric(cat[c], errors="coerce")
        m = cat["admite_mascotas"].astype(str).str.lower().str.strip().map(
            {"si":1,"sí":1,"s":1,"true":1,"1":1,"no":0,"false":0,"0":0}
        )
        cat["admite_mascotas"] = m.fillna(pd.to_numeric(cat["admite_mascotas"], errors="coerce")).fillna(0)
        cat[num_cols + ["admite_mascotas"]] = cat[num_cols + ["admite_mascotas"]].fillna(0)

        if "sitio_id" not in cat.columns:
            cat = cat.reset_index(drop=True)
            cat["sitio_id"] = np.arange(1, len(cat) + 1)
        self.catalog = cat

        # Con base en tu última traza, el pipeline también espera estas CATEGÓRICAS:
        self.CATS_REQUIRED = [
            "nacionalidad","origen","tipo_turista_preferido","compania_viaje",
            "restricciones_movilidad","nombre_sitio","tipo_sitio","accesibilidad_general",
            "idioma_info","ubicacion_geografica","clima_predominante","epoca_visita",
            # interacciones que tú generaste al entrenar:
            "x_tipoTur__tipoSit","x_epoca__tipoSit",
        ]

        # Y estas NUM/TARGET-FEATS que ya manejábamos:
        self.NUM_REQUIRED = [
            "edad","frecuencia_viaje","presupuesto_estimado","sitios_visitados",
            "calificacion_sitios_previos","tiempo_estancia_promedio",
            "costo_entrada","afluencia_promedio","duracion_esperada","admite_mascotas",
            "ratio_costo_presu","afinidad_tipo"
        ]

    # --- helpers de interacciones (ajústalos a tu lógica de entrenamiento real) ---
    def _x_tipoTur__tipoSit(self, tipo_turista: str, tipo_sitio: str) -> str:
        # ejemplo simple: concat para que el encoder lo trate como categoría
        return f"{(tipo_turista or 'na').lower()}__{(tipo_sitio or 'na').lower()}"

    def _x_epoca__tipoSit(self, epoca: str, tipo_sitio: str) -> str:
        return f"{(epoca or 'na').lower()}__{(tipo_sitio or 'na').lower()}"

    def _rows_from_user(self, user) -> pd.DataFrame:
        # num básicas
        edad = int(user.edad)
        presupuesto_estimado = _PRESUP_MAP.get(str(user.presupuesto).lower(), 50000)
        tiempo_estancia_promedio = int(user.tiempo)
        frecuencia_viaje = getattr(user, "frecuencia_viaje", 2)
        sitios_visitados = getattr(user, "sitios_visitados", 5)
        calif_prev = getattr(user, "calificacion_sitios_previos", 4.0)

        df = self.catalog.copy()

        # ---- CATEGÓRICAS de usuario repetidas por sitio ----
        df["nacionalidad"] = getattr(user, "nacionalidad", "desconocido")
        df["origen"] = getattr(user, "origen", "desconocido")
        df["tipo_turista_preferido"] = getattr(user, "tipo_turista_preferido", "desconocido")
        df["compania_viaje"] = getattr(user, "compania_viaje", getattr(user, "grupo", "desconocido"))
        df["restricciones_movilidad"] = getattr(user, "restricciones_movilidad", "ninguna")
        df["epoca_visita"] = getattr(user, "epoca_visita", "genérica")

        # ---- Num del usuario ----
        df["edad"] = edad
        df["frecuencia_viaje"] = frecuencia_viaje
        df["presupuesto_estimado"] = float(presupuesto_estimado)
        df["sitios_visitados"] = sitios_visitados
        df["calificacion_sitios_previos"] = float(calif_prev)
        df["tiempo_estancia_promedio"] = tiempo_estancia_promedio

        # ---- Derivadas esperadas por el modelo ----
        df["ratio_costo_presu"] = df["costo_entrada"].astype(float) / max(1.0, float(presupuesto_estimado))
        df["afinidad_tipo"] = df["tipo_sitio"].astype(str).apply(
            lambda t: _afinidad_por_tipo(getattr(user, "interes_ppal", ""), t)
        )

        # ---- Interacciones categóricas que el pipeline pide ----
        df["x_tipoTur__tipoSit"] = [
            self._x_tipoTur__tipoSit(tt, ts) for tt, ts in zip(df["tipo_turista_preferido"], df["tipo_sitio"])
        ]
        df["x_epoca__tipoSit"] = [
            self._x_epoca__tipoSit(ep, ts) for ep, ts in zip(df["epoca_visita"], df["tipo_sitio"])
        ]

        # --- asegurar tipos y NAs ---
        for c in self.NUM_REQUIRED:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[self.NUM_REQUIRED] = df[self.NUM_REQUIRED].fillna(0)

        for c in self.CATS_REQUIRED:
            if c not in df.columns:
                df[c] = "desconocido"
            df[c] = df[c].astype(str).fillna("desconocido")

        # Unimos todas (el pipeline de PyCaret se encargará de transformar)
        cols_finales = self.CATS_REQUIRED + self.NUM_REQUIRED
        X = df[cols_finales].copy()
        return X

    def score(self, user):
        X = self._rows_from_user(user)
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)[:, -1]
        else:
            p = self.model.predict(X)

        out = self.catalog.copy()
        out["p_like"] = p
        # Trae columnas del X necesarias para el ajuste
        out["ratio_costo_presu"] = X["ratio_costo_presu"].values
        out["afinidad_tipo"] = X["afinidad_tipo"].values

        out = self._post_adjust(out, user)
        return out

    def topn(self, user):
        df = self.score(user)
        return df.head(user.top_n)[["sitio_id","nombre_sitio","tipo_sitio","p_like","p_like_adj"]]



    def _post_adjust(self, df_ranked, user):
        """
        Ajuste ligero sobre el score del modelo para introducir preferencia de usuario.
        Sube afinidad con el tipo e introduce una pequeña penalización por costo relativo.
        """
        w_aff = 0.15   # peso afinidad (0..1)
        w_cost = 0.10  # penalización por caro vs presupuesto (0..1)

        # Por si aún no están, las tomamos del X
        if "ratio_costo_presu" not in df_ranked.columns:
            # evita división por cero
            presupuesto = max(1.0, float({"bajo":20000,"medio":50000,"alto":100000}.get(str(user.presupuesto).lower(), 50000)))
            df_ranked["ratio_costo_presu"] = df_ranked["costo_entrada"].astype(float) / presupuesto

        if "afinidad_tipo" not in df_ranked.columns:
            from math import isfinite
            df_ranked["afinidad_tipo"] = df_ranked["tipo_sitio"].astype(str).apply(
                lambda t: 1.0 if str(user.interes_ppal).lower() in t.lower() else 0.2
            )

        # Normaliza ratio a [0,1] (acotado)
        ratio_norm = df_ranked["ratio_costo_presu"].clip(0, 2.0) / 2.0
        adj = (w_aff * df_ranked["afinidad_tipo"]) - (w_cost * ratio_norm)

        base = df_ranked["p_like"] if "p_like" in df_ranked.columns else 0.5
        df_ranked["p_like_adj"] = base + adj
        return df_ranked.sort_values("p_like_adj", ascending=False, kind="mergesort")
