# app_hf.py
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# 🎨 Estilo general
st.set_page_config(page_title="Toxic Comment Detector", page_icon="🧠")

# 🚨 Advertencia idioma
with st.sidebar:
    st.title("⚠️ Info del Modelo")
    st.markdown("""
    - Este detector está basado en **BERT** fine-tuneado.
    - Actualmente solo soporta comentarios en **idioma inglés**.
    - El modelo analiza **6 categorías de toxicidad**:
        - `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
    """)

# 🧠 Título e introducción
st.title("🧠 Detector de Comentarios Tóxicos")
st.markdown("Analiza automáticamente si un comentario contiene alguna forma de toxicidad. Usa un modelo BERT entrenado para tareas multilabel.")

# 📥 Input
comentario = st.text_area("✍️ Escribe un comentario (en inglés):", height=150)

# 📦 Cargar modelo
@st.cache_resource
def cargar_modelo():
    tokenizer = BertTokenizer.from_pretrained("modelo_bert_final")
    model = BertForSequenceClassification.from_pretrained("modelo_bert_final")
    model.eval()
    return tokenizer, model

tokenizer, model = cargar_modelo()
etiquetas = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 🚀 Predicción
if st.button("🔍 Analizar Comentario"):
    if comentario.strip() == "":
        st.warning("⚠️ Por favor, escribe un comentario antes de analizar.")
    else:
        with st.spinner("Analizando..."):
            inputs = tokenizer(comentario, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).numpy()[0]
                pred = (probs > 0.5).astype(int)

        st.success("✅ Análisis completado")

        st.subheader("📊 Resultados de Toxicidad:")
        for i, clase in enumerate(etiquetas):
            if pred[i]:
                st.markdown(f"🟥 **{clase.upper()}** → ✅ Detectado  | Score: `{probs[i]:.2f}`")
            else:
                st.markdown(f"🟩 **{clase.upper()}** → ❌ No detectado | Score: `{probs[i]:.2f}`")

# 👣 Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Desarrollado por Felipe Villa (mndzk) 🧠 · Powered by HuggingFace 🤗</p>",
    unsafe_allow_html=True
)
