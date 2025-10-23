from pydantic import BaseModel
from typing import Optional

class PerfilUsuario(BaseModel):
    # numéricas ya existentes
    edad: int
    tiempo: int
    top_n: int = 5

    # tus selects previos (útiles para UI)
    presupuesto: str
    grupo: str
    movilidad: str
    interes_ppal: str

    # NUEVAS categóricas que el modelo está pidiendo:
    nacionalidad: Optional[str] = "desconocido"
    origen: Optional[str] = "desconocido"
    tipo_turista_preferido: Optional[str] = "desconocido"   # p.ej. 'nacional','internacional','eco','cultural'...
    compania_viaje: Optional[str] = "desconocido"           # 'Solo','Pareja','Familia','Grupo'...
    restricciones_movilidad: Optional[str] = "ninguna"      # 'ninguna','leve','media','alta'
    epoca_visita: Optional[str] = "genérica"                # 'Semana Santa','Navidad','Puente', etc.

    # opcionales numéricos (defaults)
    frecuencia_viaje: Optional[int] = 2
    sitios_visitados: Optional[int] = 5
    calificacion_sitios_previos: Optional[float] = 4.0


