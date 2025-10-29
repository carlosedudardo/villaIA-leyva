from pydantic import BaseModel, Field
from typing import List, Optional


class UserSurvey(BaseModel):
# Campos que vienen desde TU formulario React (mapearemos algunos)
    edad: Optional[int] = Field(None, ge=5, le=100)
    nacionalidad: Optional[str] = None
    origen: Optional[str] = None
    tipo_turista_preferido: Optional[str] = None
    compania_viaje: Optional[str] = None # solo|pareja|familia|grupo
    epoca_visita: Optional[str] = None
    presupuesto_estimado: Optional[float] = None
    restricciones_movilidad: Optional[str] = None # ninguna|leve|alta


# Campos opcionales que usa el modelo (si quieres enviarlos directos desde UI)
    clima_pref: Optional[str] = None
    tipo_sitio_pref: Optional[str] = None
    movilidad: Optional[str] = None
    presupuesto: Optional[str] = None
    tiempo_disponible_min: Optional[int] = None
    admite_mascotas: Optional[bool] = None


class Recommendation(BaseModel):
    nombre_sitio: str
    tipo_sitio: str
    score: float
    motivos: List[str] = []


class RecoResponse(BaseModel):
    items: List[Recommendation]