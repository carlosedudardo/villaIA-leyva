import React, { useState } from 'react';
import axios from 'axios';
import { MapPin, Sparkles, ChevronRight, Users, DollarSign, Heart, Info } from 'lucide-react';

const API_BASE = 'http://localhost:8000'; // Backend FastAPI

// ====================== Helpers ======================
const normalizeResponse = (data) => {
  // Acepta: [], {items: []}, {recommendations: []}
  const arr = Array.isArray(data)
    ? data
    : Array.isArray(data?.items)
    ? data.items
    : Array.isArray(data?.recommendations)
    ? data.recommendations
    : [];

  return arr.map((it) => ({
    nombre_sitio: it.nombre_sitio || it.site_name || it.nombre || 'sin_nombre',
    tipo_sitio: it.tipo_sitio || it.type || 'desconocido',
    ubicacion_geografica: it.ubicacion_geografica || it.location || 'villa_de_leyva',
    costo_entrada: Number(it.costo_entrada ?? 0),
    admite_mascotas: Boolean(it.admite_mascotas ?? 1),
    idioma_info: String(it.idioma_info || it.idiomas || 'es').slice(0, 2),
    score_like: Number(it.score ?? it.score_like ?? it.prob ?? it.proba ?? 0),
  }));
};

const dedupe = (arr) => {
  const map = {};
  for (const it of arr) {
    const k = `${it.nombre_sitio}|${it.ubicacion_geografica}`;
    if (!map[k]) map[k] = it;
  }
  return Object.values(map);
};
// =====================================================

export default function VillaLeyvaApp() {
  const [currentView, setCurrentView] = useState('home');
  const [formData, setFormData] = useState({
    edad: '',
    nacionalidad: 'Colombia',
    origen: '',
    tipo_turista_preferido: '',
    compania_viaje: '',
    epoca_visita: '',
    presupuesto_estimado: '',
    restricciones_movilidad: 'ninguna',
  });
  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [topN, setTopN] = useState(5);


  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({ ...prev, [name]: type === 'checkbox' ? checked : value }));
  };

  // ---- mapeos UI -> payload que espera el backend ----
  const mapPresupuesto = (n) => {
    const v = Number(n || 0);
    if (v <= 80000) return 'bajo';
    if (v <= 200000) return 'medio';
    return 'alto';
  };

  const mapTipoTurista = (t) => {
    const map = {
      cultural: 'museo',
      naturaleza: 'natural',
      aventura: 'aventura',
      gastronomico: 'gastronomico',
      relax_fotografia: 'plaza',
    };
    return map[t] || 'otros';
  };

  const mapMovilidad = (r) => (r === 'alta' ? 'carro' : r === 'leve' ? 'bicicleta' : 'a_pie');

  // =================== Submit ===================
  const handleSubmit = async () => {
    // Validación mínima de campos obligatorios
    if (
      !formData.edad ||
      !formData.origen ||
      !formData.tipo_turista_preferido ||
      !formData.compania_viaje ||
      !formData.epoca_visita ||
      !formData.presupuesto_estimado
    ) {
      alert('Por favor completa todos los campos obligatorios');
      return;
    }

    setIsLoading(true);
    try {
      // Payload que entiende el backend
      const payload = {
        edad: Number(formData.edad),
        // originales (por si luego se usan)
        tipo_turista_preferido: formData.tipo_turista_preferido,
        compania_viaje: formData.compania_viaje,
        presupuesto_estimado: Number(formData.presupuesto_estimado),
        restricciones_movilidad: formData.restricciones_movilidad,
        // derivados para el modelo
        presupuesto: mapPresupuesto(formData.presupuesto_estimado),
        tipo_sitio_pref: mapTipoTurista(formData.tipo_turista_preferido),
        movilidad: mapMovilidad(formData.restricciones_movilidad),
        clima_pref: 'templado_seco',
        tiempo_disponible_min: 120,
        admite_mascotas: false,
      };

      const { data } = await axios.post(`${API_BASE}/recommend?top_n=${topN}`, payload);

      console.log('RAW response:', data);

      // Normaliza + deduplica + ordena desc por score
      const normalized = normalizeResponse(data);
      const unique = dedupe(normalized).sort((a, b) => b.score_like - a.score_like);

      setRecommendations(unique);
      setCurrentView('results');
    } catch (err) {
      console.error(err);
      alert('Error obteniendo recomendaciones. ¿Está corriendo el backend en http://localhost:8000?');
    } finally {
      setIsLoading(false);
    }
  };
  // ==============================================

  // ------------------ Views ------------------
  const renderHome = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <header className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <MapPin className="text-blue-600" size={32} />
            <div>
              <h1 className="text-xl font-bold text-gray-800">Villa de Leyva</h1>
              <p className="text-xs text-gray-500">Sistema de Recomendación Turística</p>
            </div>
          </div>
          <button
            onClick={() => setCurrentView('form')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg flex items-center gap-2 transition-all duration-200 shadow-md hover:shadow-lg"
          >
            <Sparkles size={20} />
            Comenzar
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-bold text-gray-800 mb-4">Descubre Villa de Leyva</h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Sistema inteligente de recomendaciones turísticas basado en aprendizaje automático para personalizar tu experiencia
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-white p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mb-4">
              <Sparkles className="text-blue-600" size={28} />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-3">IA Avanzada</h3>
            <p className="text-gray-600">Algoritmos de aprendizaje automático entrenados con datos de preferencias turísticas</p>
          </div>

          <div className="bg-white p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mb-4">
              <Heart className="text-green-600" size={28} />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-3">Personalizado</h3>
            <p className="text-gray-600">Recomendaciones adaptadas a tus preferencias, presupuesto y estilo de viaje</p>
          </div>

          <div className="bg-white p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow">
            <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mb-4">
              <MapPin className="text-purple-600" size={28} />
            </div>
            <h3 className="text-xl font-bold text-gray-800 mb-3">Diversidad</h3>
            <p className="text-gray-600">Explora museos, plazas, arquitectura colonial, naturaleza y gastronomía</p>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-10">
          <div className="flex items-start gap-4 mb-6">
            <Info className="text-blue-600 flex-shrink-0" size={28} />
            <div>
              <h3 className="text-2xl font-bold text-gray-800 mb-4">Sobre el Proyecto</h3>
              <p className="text-gray-600 mb-4 leading-relaxed">
                Este sistema fue desarrollado por Carlos Eduardo Estupiñán Barrios y Daniel Santiago Hernández Parra de la Universidad de Boyacá.
              </p>
              <p className="text-gray-600 mb-4 leading-relaxed">
                Utilizamos técnicas de <strong>Machine Learning</strong> para analizar preferencias y generar recomendaciones personalizadas.
              </p>
              <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-600">
                <p className="text-sm text-gray-700">
                  <strong>Métricas del modelo:</strong> AUC: 0.8643 | Recall@5: 93.13% | NDCG@5: 84.38%
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="text-center mt-12">
          <button
            onClick={() => setCurrentView('form')}
            className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-12 py-4 rounded-full text-lg font-semibold flex items-center gap-3 mx-auto transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            Obtener Recomendaciones Personalizadas
            <ChevronRight size={24} />
          </button>
        </div>
      </div>
    </div>
  );

  const renderForm = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="text-center mb-8">
          <h2 className="text-4xl font-bold text-gray-800 mb-2">Cuéntanos sobre ti</h2>
          <p className="text-gray-600">Completa este breve formulario para obtener recomendaciones personalizadas</p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Users className="text-blue-600" size={24} />
              Información Personal
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Edad *</label>
                <input
                  type="number"
                  name="edad"
                  value={formData.edad}
                  onChange={handleInputChange}
                  min="1"
                  max="120"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Ej: 30"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Nacionalidad *</label>
                <select
                  name="nacionalidad"
                  value={formData.nacionalidad}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="Colombia">Colombia</option>
                  <option value="Estados Unidos">Estados Unidos</option>
                  <option value="España">España</option>
                  <option value="México">México</option>
                  <option value="Argentina">Argentina</option>
                  <option value="Otro">Otro</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Ciudad de origen *</label>
                <input
                  type="text"
                  name="origen"
                  value={formData.origen}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Ej: Bogotá"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Restricciones de movilidad</label>
                <select
                  name="restricciones_movilidad"
                  value={formData.restricciones_movilidad}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="ninguna">Ninguna</option>
                  <option value="leve">Leve</option>
                  <option value="alta">Alta</option>
                </select>
              </div>
            </div>
          </div>

          <div className="mb-8">
            <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
              <Heart className="text-blue-600" size={24} />
              Preferencias de Viaje
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Tipo de turista *</label>
                <select
                  name="tipo_turista_preferido"
                  value={formData.tipo_turista_preferido}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Selecciona una opción</option>
                  <option value="cultural">Cultural / Historia</option>
                  <option value="naturaleza">Naturaleza</option>
                  <option value="aventura">Aventura</option>
                  <option value="gastronomico">Gastronómico</option>
                  <option value="relax_fotografia">Relax / Fotografía</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Compañía de viaje *</label>
                <select
                  name="compania_viaje"
                  value={formData.compania_viaje}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Selecciona una opción</option>
                  <option value="solo">Solo</option>
                  <option value="pareja">Pareja</option>
                  <option value="familia">Familia</option>
                  <option value="grupo">Grupo de amigos</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Época de visita *</label>
                <select
                  name="epoca_visita"
                  value={formData.epoca_visita}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">Selecciona una opción</option>
                  <option value="fin_de_semana">Fin de semana</option>
                  <option value="puente_festivo">Puente festivo</option>
                  <option value="temporada_alta">Temporada alta</option>
                  <option value="temporada_baja">Temporada baja</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Presupuesto estimado (COP) *</label>
                <input
                  type="number"
                  name="presupuesto_estimado"
                  value={formData.presupuesto_estimado}
                  onChange={handleInputChange}
                  min="0"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Ej: 150000"
                />
                <div className="md:col-span-2">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Número de recomendaciones: <span className="font-semibold">{topN}</span>
                </label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  step="1"
                  value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              </div>
            </div>
          </div>

          <div className="flex gap-4 justify-between pt-6 border-t">
            <button
              onClick={() => setCurrentView('home')}
              className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
            >
              Volver
            </button>
            <button
              onClick={handleSubmit}
              disabled={isLoading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Procesando...' : 'Obtener Recomendaciones'}
              <ChevronRight size={20} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderResults = () => (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-8">
          <h2 className="text-4xl font-bold text-gray-800 mb-2">Tus Recomendaciones Personalizadas</h2>
          <p className="text-gray-600">Basado en tus preferencias, estos son los mejores lugares para ti</p>
        </div>

        {recommendations.length === 0 && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded-lg mb-6">
            No se encontraron recomendaciones con los filtros actuales. Intenta modificar tus preferencias y vuelve a intentar.
          </div>
        )}

        <div className="grid gap-6 mb-8">
          {recommendations.map((rec, index) => (
            <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold">
                      {index + 1}
                    </span>
                    <h3 className="text-2xl font-bold text-gray-800">{rec.nombre_sitio}</h3>
                  </div>
                  <div className="flex flex-wrap gap-2 mb-3">
                    <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">{rec.tipo_sitio}</span>
                    <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">{rec.ubicacion_geografica}</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-4 py-2 rounded-lg font-bold text-lg">
                    {(rec.score_like * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-gray-500 mt-1">Compatibilidad</p>
                </div>
              </div>

              <div className="grid md:grid-cols-3 gap-4 bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center gap-2">
                  <DollarSign className="text-gray-600" size={20} />
                  <div>
                    <p className="text-xs text-gray-500">Entrada</p>
                    <p className="font-semibold text-gray-800">
                      {rec.costo_entrada === 0 ? 'Gratis' : `$${rec.costo_entrada.toLocaleString()}`}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Users className="text-gray-600" size={20} />
                  <div>
                    <p className="text-xs text-gray-500">Mascotas</p>
                    <p className="font-semibold text-gray-800">{rec.admite_mascotas ? 'Sí' : 'No'}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Info className="text-gray-600" size={20} />
                  <div>
                    <p className="text-xs text-gray-500">Idiomas</p>
                    <p className="font-semibold text-gray-800">{rec.idioma_info.toUpperCase()}</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-4 justify-center">
          <button
            onClick={() => {
              setCurrentView('form');
              setRecommendations([]);
            }}
            className="px-8 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
          >
            Nueva Búsqueda
          </button>
          <button
            onClick={() => setCurrentView('home')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg transition-colors"
          >
            Volver al Inicio
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {currentView === 'home' && renderHome()}
      {currentView === 'form' && renderForm()}
      {currentView === 'results' && renderResults()}
    </>
  );
}

