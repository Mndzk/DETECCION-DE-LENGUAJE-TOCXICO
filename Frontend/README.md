# 🧠 Toxic Comment Detector with BERT + Streamlit

Este proyecto es una aplicación interactiva que detecta comentarios tóxicos utilizando un modelo BERT fine-tuneado para clasificación multilabel.

## 📌 Descripción
El modelo analiza textos en **idioma inglés** y clasifica los comentarios según seis categorías de toxicidad:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

## 🧠 Tecnologías utilizadas

- HuggingFace Transformers
- PyTorch
- Streamlit
- Scikit-learn

## 🚀 Cómo ejecutar la aplicación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/toxic-comment-detector.git
cd toxic-comment-detector
```

### 2. Instalar dependencias
```bash
pip install -r frontend/requirements.txt
```

### 3. Ejecutar la app
```bash
cd frontend
streamlit run app_hf.py
```

⚠️ Asegúrate de tener la carpeta `modelo_bert_final/` dentro del mismo directorio de `app.py`.

## 📊 Resultados del modelo (F1-score promedio)

| Modelo         | F1 promedio |
|----------------|-------------|
| Logistic       | 0.194       |
| Random Forest  | 0.260       |
| LinearSVC      | 0.370       |
| Red Neuronal   | 0.454       |
| **BERT (final)**| **0.604**  ✅

## 📁 Estructura del proyecto

```
toxic-comment-detector/
├── backend/
│   ├── model_training.ipynb
│   └── modelo_bert_final/
├── frontend/
│   ├── app.py
│   ├── requirements.txt
├── data/
│   └── train.csv
├── results/
│   └── tabla_comparativa_final.csv
├── README.md
├── LICENSE
└── .gitignore
```

## 🧑‍💻 Autor
Felipe Villa · Desarrollador de modelos NLP y apps interactivas de IA.
