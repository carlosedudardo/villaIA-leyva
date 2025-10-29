# app/core/feature_map.py
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# === Mapas UI → features del modelo ===
MAP_PRESUP = lambda v: ('bajo' if (v is not None and float(v) <= 80_000)
                        else 'medio' if (v is not None and float(v) <= 200_000)
                        else 'alto')

MAP_TIPO_TURISTA = {
    'cultural': 'museo',
    'naturaleza': 'natural',
    'aventura': 'aventura',
    'gastronomico': 'gastronomico',
    'relax_fotografia': 'plaza',
}

MAP_MOVILIDAD = {
    'ninguna': 'a_pie',
    'leve': 'bicicleta',
    'alta': 'carro',
}

MAP_COMPANIA = {
    'solo': 'solo',
    'pareja': 'pareja',
    'familia': 'familia',
    'grupo': 'amigos',
}

def build_pair_matrix(catalog_df: pd.DataFrame, ui: Dict) -> pd.DataFrame:
    """
    Devuelve el catálogo completo + columnas de UI y derivadas.
    No recorta columnas: PyCaret necesita ver todas las que vio en entrenamiento.
    """
    df = catalog_df.copy()
    df = df.replace({pd.NA: np.nan})
    # Derivados desde la UI
    presupuesto = ui.get('presupuesto') or MAP_PRESUP(ui.get('presupuesto_estimado'))
    tipo_pref   = ui.get('tipo_sitio_pref') or MAP_TIPO_TURISTA.get(ui.get('tipo_turista_preferido'), 'otros')
    movilidad   = ui.get('movilidad') or MAP_MOVILIDAD.get(ui.get('restricciones_movilidad'), 'a_pie')
    compania    = ui.get('compania') or MAP_COMPANIA.get(ui.get('compania_viaje') or 'solo', 'solo')

    # Variables “crudas” que probablemente existían en el dataset de entrenamiento
    df['nacionalidad']             = ui.get('nacionalidad', 'Colombia')
    df['origen']                   = ui.get('origen', '')
    df['tipo_turista_preferido']   = ui.get('tipo_turista_preferido', '')
    df['compania_viaje']           = ui.get('compania_viaje', 'solo')
    df['epoca_visita']             = ui.get('epoca_visita', '')
    df['restricciones_movilidad']  = ui.get('restricciones_movilidad', 'ninguna')
    df['presupuesto_estimado']     = ui.get('presupuesto_estimado', np.nan)

    # Features que usa el pipeline final (ajústalas si hace falta)
    df['edad']                   = ui.get('edad', np.nan)
    df['compania']               = compania
    df['presupuesto']            = presupuesto
    df['movilidad']              = movilidad
    df['tipo_sitio_pref']        = tipo_pref
    df['tiempo_disponible_min']  = ui.get('tiempo_disponible_min', 120)
    df['clima_pref']             = ui.get('clima_pref', 'templado_seco')
    df['admite_mascotas_user']   = bool(ui.get('admite_mascotas', False))

    # Si el catálogo tiene esta columna, construimos un “match”
    if 'admite_mascotas' in df.columns:
        df['match_mascotas'] = (df['admite_mascotas'].astype(bool) == df['admite_mascotas_user']).astype(int)

    return df

# === Lista de columnas esperadas por el ESTIMADOR FINAL ===
# ¡Ajusta esta lista para que coincida con tu pipeline/modelo entrenado!
# === Lista de columnas esperadas por el ESTIMADOR FINAL ===
MODEL_FEATURES = [
    'tipo_sitio','clima_predominante','admite_mascotas','accesibilidad_general',
    'duracion_esperada','costo_entrada','compania','presupuesto','movilidad',
    'tiempo_disponible_min','match_mascotas','edad','tipo_sitio_pref',
]


# === Columnas “crudas” que PyCaret vio en entrenamiento (de tu error)
LIKELY_TRAIN_COLS = [
    'nacionalidad','origen','tipo_turista_preferido','compania_viaje',
    'nombre_sitio','ubicacion_geografica','epoca_visita','restricciones_movilidad',
    'idioma_info','x_tipoTur__tipoSit','x_epoca__tipoSit','presupuesto_estimado',
    'frecuencia_viaje','sitios_visitados','calificacion_sitios_previos',
    'tiempo_estancia_promedio','afluencia_promedio','ratio_costo_presu','afinidad_tipo',
    # Nota: si tu catálogo tiene numéricos como 'costo_entrada'/'duracion_esperada' en el crudo, PyCaret puede verlos también.
    'costo_entrada','duracion_esperada','edad','tiempo_disponible_min'
]

TRAIN_NUMERIC_COLS = {
    'presupuesto_estimado','frecuencia_viaje','sitios_visitados','calificacion_sitios_previos',
    'tiempo_estancia_promedio','afluencia_promedio','ratio_costo_presu',
    'costo_entrada','duracion_esperada','edad','tiempo_disponible_min','match_mascotas'
}


def ensure_training_columns(df: pd.DataFrame, add_extra: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Asegura columnas crudo-entrenamiento para PyCaret:
    - Crea faltantes con NaN si son numéricas, '' si categóricas.
    - Convierte a numérico (to_numeric) las columnas NUMÉRICAS y coerciona ''→NaN.
    """
    df = df.copy()
    candidates = set(LIKELY_TRAIN_COLS) | set(add_extra or [])

    for col in candidates:
        if col not in df.columns:
            if col in TRAIN_NUMERIC_COLS:
                df[col] = np.nan
            else:
                df[col] = ''  # categórica por defecto

    # Coerciona numéricas a float (''/pd.NA -> np.nan)
    for col in TRAIN_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Convierte columnas "string dtype" a object para evitar NAType aguas abajo
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c].dtype):
            df[c] = df[c].astype('object')


    return df

def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena y ordena las features finales que espera el estimador.
    """
    df = df.copy()

    numeric_cols = ['costo_entrada','duracion_esperada','tiempo_disponible_min','edad','match_mascotas']
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Puedes dejar NaN y permitir que el imputer del pipeline actúe,
    # o rellenar a 0 aquí si tu estimador puro sklearn lo requiere.
    # df[numeric_cols] = df[numeric_cols].fillna(0)

    for c in MODEL_FEATURES:
        if c not in df.columns:
            df[c] = ''
    cat_cols = [c for c in MODEL_FEATURES if c not in numeric_cols]
    for c in cat_cols:
        df[c] = df[c].astype('string')

    return df[MODEL_FEATURES]
