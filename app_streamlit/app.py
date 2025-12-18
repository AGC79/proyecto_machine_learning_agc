from PIL import Image
import streamlit as st
import base64
from io import BytesIO
import random
import funciones as f

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="Dog Emotion Classifier",
    page_icon="🐶",
    layout="centered"
)

# ---------------------- CSS ----------------------
st.markdown("""
<style>
/* ===== FONDO ===== */
.stApp {
    background: linear-gradient(135deg, #d9a184, #f0bfa1);
    font-family: 'Segoe UI', sans-serif;
}

/* ===== TÍTULO ===== */
h1 {
    margin: 20px 0 15px 0;
    font-size: 42px;
    font-weight: 700;
    color: #3a1f1a;
    text-align: center;
    line-height: 1.1;
}

/* ===== GIF CIRCULAR ===== */
.gif-container {
    width: 160px;
    height: 160px;
    margin: 0 auto 20px auto;
    border-radius: 50%;
    overflow: hidden;
    border: 3px solid rgba(255,255,255,0.7);
}

/* ===== BARRA DE TEXTO ===== */
.text-bar {
    width: 100%;
    padding: 6px 0;
    background-color: rgba(255,255,255,0.3);
    text-align: center;
    font-weight: 600;
    color: #3a1f1a;
    border-radius: 6px;
    margin-bottom: 16px;
}

/* ===== BOTÓN ===== */
div.stButton > button {
    background: linear-gradient(135deg, #c0392b, #e17055);
    color: white;
    font-size: 18px;
    font-weight: bold;
    height: 3em;
    width: 100%;
    border-radius: 12px;
    border: none;
}

/* ===== INDICADORES ===== */
.indicator {
    width: 105px;
    height: 105px;
    border-radius: 50%;
    background-color: #b79a92;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: white;
    margin: auto;
    transition: all 0.35s ease;
}

.active {
    background-color: #27ae60;
    box-shadow: 0 0 18px rgba(39,174,96,0.85);
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    font-size: 13px;
    color: #4a2b25;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- TÍTULO ----------------------
st.markdown("<h1>Clasificador de Emociones Caninas</h1>", unsafe_allow_html=True)

# ---------------------- GIF ----------------------
def load_gif(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

gif_base64 = load_gif("img/perro.gif")
st.markdown(
    f"<div class='gif-container'>"
    f"<img src='data:image/gif;base64,{gif_base64}' style='width:100%; height:100%; object-fit:cover;'/>"
    "</div>",
    unsafe_allow_html=True
)

# ---------------------- BARRA DE TEXTO ----------------------
st.markdown(
    "<div class='text-bar'>Sube una imagen de un perro y predice su emoción usando Machine Learning</div>",
    unsafe_allow_html=True
)

# ---------------------- IMAGE UPLOAD ----------------------
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Convertir la imagen a base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Mostrar la imagen centrada y más pequeña
    st.markdown(
        f"<div style='text-align:center; margin-bottom:20px;'>"
        f"<img src='data:image/png;base64,{img_str}' width='300' style='border-radius:12px;'/>"
        f"</div>",
        unsafe_allow_html=True
    )
    img_3d, img_flat = f.procesar_imagen(uploaded_file)

# ---------------------- IMAGE PROCESSING ----------------------

# check

# ---------------------- MODEL SELECTION ----------------------
st.markdown("<div class='text-bar'>🧠 Selección de modelo</div>", unsafe_allow_html=True)

modelo = st.selectbox(
    "",
    ["RandomForest", "KNN", "Red Neuronal", "Red Convolucional + Red Neuronal", 
     "Red Convolucional + Red Neuronal + Data Augmentation"],
    label_visibility="collapsed"
)
st.title(modelo)
# ---------------------- PREDICCIÓN ----------------------
# Botón centrado usando columnas
col1, col2, col3, col4 = st.columns([1,1,2,1])
with col3:
    predict = st.button("Predecir emoción")

emociones = ["angry", "happy", "relaxed", "sad"]
prediccion = None

if predict and uploaded_file:
    if modelo == "RandomForest":
        prediccion = f.rf_model(img_flat)
    elif modelo == "KNN":
        prediccion = f.knn_model(img_flat)
    elif modelo == "Red Neuronal":
        prediccion = f.rn_model(img_3d)
    elif modelo == "Red Convolucional + Red Neuronal":
        prediccion = f.conv_rn_model(img_3d)
    elif modelo == "Red Convolucional + Red Neuronal + Data Augmentation":
        prediccion = f.conv_rn_da_model(img_3d)

elif predict:
    st.error("❌ Debes cargar una imagen primero")

# ---------------------- INDICADORES ----------------------
st.markdown("<div class='text-bar'>📊 Emoción detectada</div>", unsafe_allow_html=True)

cols = st.columns(4)
for col, emocion in zip(cols, emociones):
    active = "active" if emocion == prediccion else ""
    col.markdown(
        f"<div class='indicator {active}'>{emocion.upper()}</div>",
        unsafe_allow_html=True
    )

# ---------------------- FOOTER ----------------------
st.markdown("<div class='footer'>🔬 Demo visual – Integrable con modelos reales de Deep Learning</div>", unsafe_allow_html=True)
