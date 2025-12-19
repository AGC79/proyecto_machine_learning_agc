from PIL import Image
import streamlit as st
import base64
from io import BytesIO
import funciones as f

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Dog Emotion Classifier", page_icon="🐶", layout="centered")

# ---------------------- CSS ----------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #d9a184, #f0bfa1); font-family: 'Segoe UI', sans-serif; }
h1 { margin: 20px 0 15px 0; font-size: 42px; font-weight: 700; color: #3a1f1a; text-align: center; }
.gif-container { width: 160px; height: 160px; margin: 0 auto 20px auto; border-radius: 50%; overflow: hidden; border: 3px solid rgba(255,255,255,0.7); }
.text-bar { width: 100%; padding: 6px 0; background-color: rgba(255,255,255,0.3); text-align: center; font-weight: 600; color: #3a1f1a; border-radius: 6px; margin-bottom: 16px; }
div.stButton > button { background: linear-gradient(135deg, #c0392b, #e17055); color: white; font-size: 18px; font-weight: bold; width: 100%; border-radius: 12px; border: none; }
.indicator { width: 100px; height: 100px; border-radius: 50%; background-color: #b79a92; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; margin: auto; transition: all 0.35s ease; margin-bottom: 10px; }
.active { background-color: #27ae60; box-shadow: 0 0 18px rgba(39,174,96,0.85); }
.prob-text { text-align: center; font-weight: bold; color: #3a1f1a; margin-bottom: 2px; font-size: 14px; }
.certeza-label { text-align: center; font-size: 11px; color: #3a1f1a; font-style: italic; margin-top: -5px; }
.footer { text-align: center; font-size: 13px; color: #4a2b25; margin-top: 30px; border-top: 1px solid rgba(0,0,0,0.1); padding-top: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Clasificador de Emociones Caninas</h1>", unsafe_allow_html=True)

# ---------------------- GIF ----------------------
def load_gif(path):
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

try:
    gif_base64 = load_gif("img/perro.gif")
    st.markdown(f"<div class='gif-container'><img src='data:image/gif;base64,{gif_base64}' style='width:100%; height:100%; object-fit:cover;'/></div>", unsafe_allow_html=True)
except:
    st.info("🐶 (GIF no encontrado)")

st.markdown("<div class='text-bar'>Sube una imagen de un perro y predice su emoción</div>", unsafe_allow_html=True)

# ---------------------- UPLOAD & PREPROCESS ----------------------
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg","jpeg","png"], label_visibility="collapsed")
img_3d, img_flat = None, None

if uploaded_file:
    image = Image.open(uploaded_file)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(f"<div style='text-align:center; margin-bottom:20px;'><img src='data:image/png;base64,{img_str}' width='250' style='border-radius:12px;'/></div>", unsafe_allow_html=True)
    img_3d, img_flat = f.procesar_imagen(uploaded_file)

# ---------------------- MODEL SELECTION ----------------------
st.markdown("<div class='text-bar'>🧠 Selección de modelo</div>", unsafe_allow_html=True)
modelo_name = st.selectbox("", ["RandomForest", "KNN", "Red Neuronal", "Red Convolucional + Red Neuronal", "Red Convolucional + Red Neuronal + Data Augmentation"], label_visibility="collapsed")

col1, col2, col3, col4 = st.columns([1,1,2,1])
with col3:
    predict_btn = st.button("Predecir emoción")

# ---------------------- LOGIC ----------------------
emociones = ["angry", "happy", "relaxed", "sad"]
prediccion = None
probabilidades = None

if predict_btn and uploaded_file:
    if modelo_name == "RandomForest":
        prediccion, probabilidades = f.rf_model(img_flat)
    elif modelo_name == "KNN":
        prediccion, probabilidades = f.knn_model(img_flat)
    elif modelo_name == "Red Neuronal":
        prediccion, probabilidades = f.rn_model(img_3d)
    elif modelo_name == "Red Convolucional + Red Neuronal":
        prediccion, probabilidades = f.conv_rn_model(img_3d)
    elif modelo_name == "Red Convolucional + Red Neuronal + Data Augmentation":
        prediccion, probabilidades = f.conv_rn_da_model(img_3d)
elif predict_btn:
    st.error("❌ Debes cargar una imagen primero")

# ---------------------- RESULTS ----------------------
st.markdown("<div class='text-bar'>📊 Resultados del Análisis</div>", unsafe_allow_html=True)

cols = st.columns(4)
for i, (col, emo) in enumerate(zip(cols, emociones)):
    is_active = "active" if emo == prediccion else ""
    # Círculo visual
    col.markdown(f"<div class='indicator {is_active}'>{emo.upper()}</div>", unsafe_allow_html=True)
    
    # Barra de Probabilidad con Nivel de Certeza
    if probabilidades is not None:
        p_val = float(probabilidades[i])
        porcentaje = p_val * 100
        
        # Lógica de etiqueta de Nivel de Certeza
        if porcentaje >= 80:
            certeza_txt = "Nivel de Certeza: Alto"
        elif porcentaje >= 40:
            certeza_txt = "Nivel de Certeza: Medio"
        else:
            certeza_txt = "Nivel de Certeza: Bajo"
            
        col.markdown(f"<div class='prob-text'>{porcentaje:.1f}%</div>", unsafe_allow_html=True)
        col.progress(p_val)
        col.markdown(f"<p class='certeza-label'>{certeza_txt}</p>", unsafe_allow_html=True)

st.markdown(f"<div class='footer'>🔬 Tecnología de Predicción 2025 – By Álvaro Guerra Cabello.</div>", unsafe_allow_html=True)
