# 🌱 Clasificador Automático de ODS

Clasificador de textos libres (propuestas, proyectos o artículos) para identificar a qué **Objetivo de Desarrollo Sostenible (ODS)** pertenecen, utilizando técnicas de NLP y Machine Learning.

---

## 📁 Estructura del Proyecto

```
Microproyecto_2/
├── dataset/                        # Datos traducidos y aumentados de OSDG
├── notebooks/
│   └── microproyecto2.ipynb        # Cuaderno principal documentado
├── models/
│   └── text_ods_pipeline.joblib    # Pipeline entrenado y guardado
├── contexto/                       # (No se sube a Git) Notas, borradores, recursos locales
├── app.py                          # (Opcional) Aplicación interactiva en Streamlit
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Este archivo
```

---

## 🚀 Instalación y Uso

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Ingresar un texto** en la interfaz y presionar **"Clasificar ODS"** para obtener la predicción.

---

## 🛠️ Tecnologías

- **Python** — Lenguaje principal
- **Streamlit** — Interfaz web interactiva
- **scikit-learn** — Pipeline de clasificación (TF-IDF + modelo)
- **NLTK** — Preprocesamiento de texto en español (tokenización, stopwords, stemming)
- **joblib** — Serialización del modelo entrenado
