
import streamlit as st
import joblib
import nltk
import glob
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# 1. Descarga de recursos NLTK necesarios
nltk.download('stopwords', quiet=True)

# 2. Configuración de la interfaz
st.set_page_config(
    page_title="Clasificador de ODS",
    page_icon="🌱",
    layout="centered"
)

# 3. Función de preprocesamiento de texto idéntica a la del entrenamiento
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(str(text).lower())
    spanish_stopwords = set(stopwords.words('spanish'))
    tokens = [token for token in tokens if token not in spanish_stopwords]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

# 4. Función para reensamblar y cargar el pipeline
@st.cache_resource
def cargar_recursos():
    # Reensamblar las partes del pipeline si no existe el archivo completo
    if not os.path.exists('modelo_ods_pipeline.pkl'):
        partes = sorted(glob.glob('modelo_ods_pipeline.pkl.part*'))
        if partes:
            with open('modelo_ods_pipeline.pkl', 'wb') as outfile:
                for parte in partes:
                    with open(parte, 'rb') as infile:
                        outfile.write(infile.read())
                        
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

texto_usuario = st.text_area(
    "Texto a analizar:",
    height=180,
    placeholder="Ejemplo: Implementación de sistemas de energía solar fotovoltaica para comunidades rurales..."
)

if st.button("Clasificar ODS", type="primary"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingrese un texto antes de presionar el botón.")
    else:
        ods_predicho = pipeline.predict([texto_usuario])[0]
        probabilidades = pipeline.predict_proba([texto_usuario])[0]
        clases = list(pipeline.classes_)
        idx_clase = clases.index(ods_predicho)
        confianza = probabilidades[idx_clase] * 100

        nombre_ods = ODS_NOMBRES.get(ods_predicho, f"ODS {ods_predicho}")

        st.success("### Resultado de la Clasificación")
        st.subheader(f"🎯 {nombre_ods}")
        st.metric(label="Nivel de Confianza", value=f"{confianza:.2f}%")

        if confianza < 30.0:
            st.info("⚠️ La confianza de la predicción es baja (<30%). El texto podría abarcar múltiples temáticas o ser ambiguo.")
