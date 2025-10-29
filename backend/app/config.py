from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# OJO: sin extensión .pkl
MODEL_PATH = ROOT_DIR / "models" / "modelo_cls_like_v3"
CATALOG_PATH = BASE_DIR / "data" / "catalogo_vdl_lugares_unico.csv"
TOP_N_DEFAULT = 5
