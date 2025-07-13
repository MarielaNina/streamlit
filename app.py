import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Configuración de página
st.set_page_config(
    page_title="Traductor Español - Aymara", 
    page_icon="🌎",
    layout="centered"
)

# Google Fonts + CSS personalizado más moderno
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }

    .main-header {
        text-align: center;
        color: #6B5299; /* Morado grisáceo oscuro */
        font-size: 2.5rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #8B7CA6; /* Morado grisáceo medio */
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .stTextArea textarea {
        background-color: #F7F3FA; /* Fondo casi blanco con tinte lavanda */
        border-radius: 8px;
    }

    .result-container {
        background-color: #E9E2F8; /* Lila muy pálido */
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1.5rem;
        border-left: 4px solid #A78BC8; /* Morado pastel claro */
    }

    .success-text {
        color: #A78BC8; /* Morado pastel claro */
        font-weight: 500;
        font-size: 1.1rem;
    }

    .stButton>button {
        background-color: #A78BC8; /* Morado pastel claro */
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-size: 1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #9575CD; /* Morado pastel más oscuro (hover) */
    }

    .footer {
        text-align: center;
        color: #8B7CA6; /* Morado grisáceo medio */
        font-size: 0.85rem;
        margin-top: 2rem;
    }
</style>

""", unsafe_allow_html=True)

# Encabezado
st.markdown('<h1 class="main-header">Traduce del Español al Aymara</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Introduce tu texto y obtén la traducción al instante</p>', unsafe_allow_html=True)

# Validación de letras repetidas
def validar_letras_repetidas(texto):
    patron = r'(.)\1{3,}'
    return not bool(re.search(patron, texto))

# Cargar modelo
@st.cache_resource
def load_model():
    model_name = "MarielaNina/nllb-200-600-spanish-aimara-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Historial
if 'historial' not in st.session_state:
    st.session_state.historial = []

# Entrada
texto = st.text_area("Ingresa el texto:", placeholder="Ejemplo: Buenos días, ¿cómo estás?")

# Botón traducir
if st.button("Traducir", use_container_width=True):
    if texto.strip():
        texto_limpio = texto.strip()
        if len(texto_limpio) < 4:
            st.error("El texto debe tener al menos 4 caracteres.")
        elif not validar_letras_repetidas(texto_limpio):
            st.error("El texto contiene demasiadas letras repetidas.")
        elif texto_limpio.lower() in [t['original'].lower() for t in st.session_state.historial]:
            st.warning("Este texto ya fue traducido.")
        else:
            inputs = tokenizer(texto_limpio, return_tensors="pt")
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("ayr_Latn")
            with st.spinner("Traduciendo..."):
                outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
            traduccion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if traduccion.strip() and traduccion.lower() != texto_limpio.lower():
                st.session_state.historial.append({'original': texto_limpio, 'traduccion': traduccion})
                st.markdown('<div class="result-container"><strong class="success-text">Traducción:</strong><br>' + traduccion + '</div>', unsafe_allow_html=True)
            else:
                st.error("No se pudo traducir correctamente.")
    else:
        st.warning("Ingresa un texto para traducir.")

# Historial
if st.session_state.historial:
    st.markdown("---")
    st.markdown("### Historial")
    for traduccion in reversed(st.session_state.historial[-3:]):
        st.write(f"**{traduccion['original']}** → *{traduccion['traduccion']}*")
    if st.button("Limpiar historial"):
        st.session_state.historial = []
        st.rerun()

# Footer
st.markdown('<p class="footer"> Traductor Español → Aymara | Desarrollado por Mariela Nina</p>', unsafe_allow_html=True)

