import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Traductor Aymara ↔ Español", page_icon="🌍")

st.title("🌍 Traductor Aymara ↔ Español")
st.write("Traduce textos entre Español y Aymara usando tu modelo fine-tuned de Hugging Face.")

@st.cache_resource
def load_model():
    model_name = "MarielaNina/nllb-200-600-spanish-aimara-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

direccion = st.selectbox(
    "Selecciona la dirección de traducción:",
    ("Español → Aymara", "Aymara → Español")
)

texto = st.text_area("Ingresa el texto a traducir:")

if st.button("Traducir"):
    if texto.strip():
        inputs = tokenizer(texto, return_tensors="pt")

        # Define forced_bos_token_id según dirección
        if direccion == "Español → Aymara":
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("ayr_Latn")
        else:
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("spa_Latn")

        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id
        )

        traduccion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.write("**Traducción:**")
        st.success(traduccion)
    else:
        st.warning("Por favor ingresa un texto para traducir.")
