import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Traductor Aymara ↔ Español", page_icon="🌍")

st.title("🌍 Traductor Automático")
st.write("Traduce texto entre Aymara y Español con tu modelo personalizado.")

# --- Carga del modelo ---
@st.cache_resource
def load_model():
    model_name = "TU_USUARIO/TU_MODELO"  # Ruta a tu modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    translator = pipeline("translation", model=model, tokenizer=tokenizer)
    return translator

translator = load_model()

# --- Interfaz de usuario ---
input_text = st.text_area("Introduce el texto:", "")

if st.button("Traducir"):
    if input_text.strip():
        resultado = translator(input_text)
        st.success(resultado[0]['translation_text'])
    else:
        st.warning("Por favor ingresa un texto.")
