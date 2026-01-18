import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import spacy
import pandas as pd
from pytesseract import Output

# Configuration Tesseract
# Note : Assurez-vous que ce chemin est correct pour votre installation

# Chargement du modèle NLP français
@st.cache_resource
def load_nlp():
    try:
        return spacy.load("fr_core_news_sm")
    except:
        return None

nlp = load_nlp()

# Configuration de la page
st.set_page_config(
    page_title="FABET TRANSCRIPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour le Dark Mode
st.markdown("""
<style>
    /* Global Background and Text Color */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Police moderne */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #FFFFFF !important;
    }
    
    /* Titres en Blanc */
    h1, h2, h3, h4, p, label {
        color: #FFFFFF !important;
    }
    
    /* Cartes Dark */
    .card {
        background: #1E1E1E;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333333;
        margin-bottom: 1.5rem;
    }
    
    /* Barre latérale sombre */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333333;
    }
    
    /* Inputs et Text Areas */
    .stTextArea textarea {
        background-color: #262730 !important;
        color: #FFFFFF !important;
        border: 1px solid #444 !important;
    }
    
    /* Métriques */
    .metric-box {
        background: #262730;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #444;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00FFCC; /* Couleur d'accent pour la précision */
    }

    /* Séparateurs */
    .divider {
        height: 1px;
        background: #333333;
        margin: 2rem 0;
    }
    
    /* Style des boutons */
    .stButton button {
        background: #FFFFFF;
        color: #000000 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.markdown("<h1 style='margin-bottom: 0.5rem;'>FABET TRANSCRIPT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity: 0.8; font-size: 1.1rem;'>Extraction OCR et analyse de documents haute précision</p>", unsafe_allow_html=True)

with col_logo:
    st.markdown("<div style='text-align: right; font-size: 2rem; font-weight: bold;'>FT</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Barre latérale
with st.sidebar:
    st.markdown("### Configuration")
    preprocess = st.checkbox("Amélioration d'image", value=True)
    st.markdown("---")
    st.markdown("### Informations techniques")
    st.markdown("- **Format:** JPG, PNG")
    st.markdown("- **Langue:** Français")
    st.markdown("- **Moteur:** Tesseract OCR")

# Zone d'importation
st.markdown("<h3>Importation du document</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight: 600; margin-bottom: 1rem;'>Image originale</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Prétraitement
    if preprocess:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    else:
        processed_img = img_array

    # OCR avec calcul de précision
    with st.spinner('Analyse OCR en cours...'):
        # Extraction des données détaillées pour obtenir la confiance (precision)
        data = pytesseract.image_to_data(processed_img, lang='fra', output_type=Output.DATAFRAME)
        # Nettoyage des données (enlever les confiances de -1 qui correspondent aux blocs de mise en page)
        valid_conf = data[data.conf != -1]
        
        if not valid_conf.empty:
            avg_confidence = valid_conf.conf.mean()
            text = pytesseract.image_to_string(processed_img, lang='fra')
        else:
            avg_confidence = 0
            text = ""

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight: 600; margin-bottom: 1rem;'>Résultats de la transcription</p>", unsafe_allow_html=True)
        
        # Affichage des métriques incluant la PRÉCISION
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-box'><p style='font-size: 0.8rem; opacity: 0.7;'>Précision</p><p class='metric-value'>{avg_confidence:.1f}%</p></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-box'><p style='font-size: 0.8rem; opacity: 0.7;'>Mots</p><p class='metric-value'>{len(text.split())}</p></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-box'><p style='font-size: 0.8rem; opacity: 0.7;'>Caractères</p><p class='metric-value'>{len(text)}</p></div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.text_area("Texte extrait", text, height=250, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    # Analyse NLP si du texte existe
    if text.strip() and nlp:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h3>Analyse Intelligente</h3>", unsafe_allow_html=True)
        
        doc = nlp(text)
        tab1, tab2 = st.tabs(["💎 Entités Nommées", "📊 Statistiques"])
        
        with tab1:
            entities = [{"Entité": ent.text, "Type": ent.label_} for ent in doc.ents]
            if entities:
                st.table(pd.DataFrame(entities))
            else:
                st.write("Aucune entité détectée.")
                
        with tab2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.write("**Mots les plus fréquents**")
                words = [t.text.lower() for t in doc if not t.is_stop and not t.is_punct and len(t.text) > 1]
                if words:
                    st.bar_chart(pd.Series(words).value_counts().head(10))
            with col_s2:
                st.write("**Répartition linguistique**")
                pos_counts = pd.Series([t.pos_ for t in doc if t.pos_ not in ['PUNCT', 'SPACE']]).value_counts()
                st.dataframe(pos_counts, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Actions de sortie
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    c_down1, c_down2 = st.columns(2)
    with c_down1:
        st.download_button("📥 Télécharger la transcription (TXT)", text, file_name="transcript.txt")
    with c_down2:
        if st.button("🔄 Nouvelle analyse"):
            st.rerun()

else:
    # État vide
    st.markdown(f"""
    <div style='text-align: center; padding: 5rem 2rem; border: 2px dashed #333; border-radius: 20px;'>
        <h2 style='color: #FFFFFF;'>Prêt pour l'extraction</h2>
        <p style='color: #AAAAAA;'>Glissez-déposez un document pour commencer l'analyse OCR avec calcul de précision.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div style='text-align: center; color: #555; font-size: 0.8rem; margin-top: 5rem;'>FABET TRANSCRIPT © 2024 - Système d'analyse sécurisé</div>", unsafe_allow_html=True)