
import streamlit as st
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# 1. Descarga de recursos NLTK necesarios para el preprocesamiento
nltk.download('stopwords', quiet=True)

# 2. Configuración de la interfaz
st.set_page_config(
    page_title="Clasificador de ODS",
    page_icon="🌱",
    layout="centered"
)

# 3. Función de preprocesamiento de texto (Nombre e implementación exactos al cuaderno)
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(str(text).lower())
    spanish_stopwords = set(stopwords.words('spanish'))
    tokens = [token for token in tokens if token not in spanish_stopwords]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# 4. Cargar el pipeline y el diccionario de nombres
@st.cache_resource
def cargar_recursos():
    pipeline = joblib.load('modelo_ods_pipeline.pkl')
    ods_nombres = joblib.load('ods_nombres.pkl')
    return pipeline, ods_nombres

try:
    pipeline, ODS_NOMBRES = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar los archivos .pkl: {e}")
    st.stop()

# 5. Interfaz interactiva
st.title("🌱 Clasificador Automático de ODS")
st.write(
    "Ingrese un texto libre (propuesta, proyecto o artículo) para identificar "
    "a qué **Objetivo de Desarrollo Sostenible (ODS)** pertenece."
)

st.markdown("---")

# Entrada de texto del usuario
texto_usuario = st.text_area(
    "Texto a analizar:",
    height=180,
    placeholder="Ejemplo: Implementación de sistemas de energía solar fotovoltaica para comunidades rurales..."
)

if st.button("Clasificar ODS", type="primary"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingrese un texto antes de presionar el botón.")
    else:
        # Nota: Pasamos 'texto_usuario' directamente en una lista. 
        # El pipeline ejecuta 'preprocess_text' internamente dentro del TfidfVectorizer.
        ods_predicho = pipeline.predict([texto_usuario])[0]
        
        # Cálculo de probabilidades y confianza
        probabilidades = pipeline.predict_proba([texto_usuario])[0]
        clases = list(pipeline.classes_)
        idx_clase = clases.index(ods_predicho)
        confianza = probabilidades[idx_clase] * 100

        # Mapeo del nombre del ODS
        nombre_ods = ODS_NOMBRES.get(ods_predicho, f"ODS {ods_predicho}")

        # Despliegue de resultados
        st.success("### Resultado de la Clasificación")
        st.subheader(f"🎯 {nombre_ods}")
        st.metric(label="Nivel de Confianza", value=f"{confianza:.2f}%")

        if confianza < 30.0:
            st.info("⚠️ La confianza de la predicción es baja (<30%). El texto podría abarcar múltiples temáticas o ser ambiguo.")
