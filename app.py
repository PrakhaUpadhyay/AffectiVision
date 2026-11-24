import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# -----------------------------------------------------------------------------
# Page Configuration (No Sidebar — Full-Width Web App)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AffectiVision — AI Facial Emotion Scanner",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join("outputs", "emotion_cnn_model.keras")
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Bright red in RGB for the face bounding box
RED_BOX_RGB = (255, 0, 0)

EMOTION_PALETTE = {
    "Happy":    {"hex": "#4ade80", "light": "#e1f5ee", "dark": "rgba(74, 222, 128, 0.14)"},
    "Surprise": {"hex": "#f59e0b", "light": "#fff7ed", "dark": "rgba(245, 158, 11, 0.14)"},
    "Neutral":  {"hex": "#6c4ff6", "light": "#f3e8ff", "dark": "rgba(108, 79, 246, 0.16)"},
    "Sad":      {"hex": "#3b82f6", "light": "#e6f1fb", "dark": "rgba(59, 130, 246, 0.14)"},
    "Angry":    {"hex": "#f43f5e", "light": "#fef2f2", "dark": "rgba(244, 63, 94, 0.15)"},
    "Fear":     {"hex": "#e12afb", "light": "#fce7f3", "dark": "rgba(225, 42, 251, 0.14)"},
    "Disgust":  {"hex": "#8b5cf6", "light": "#e6fffa", "dark": "rgba(139, 92, 246, 0.14)"},
}

TRANSLATIONS = {
    "English": {
        "nav_scan": "Scanner",
        "nav_stats": "Analytics",
        "nav_about": "About",
        "nav_faq": "FAQs",
        "login": "Sign In",
        "signup": "Create Account",
        "logout": "Sign Out",
        "pill": "AI FACIAL EXPRESSION SCANNER",
        "hero_h1": "Understand emotions in every face",
        "hero_p": "Upload a photo or capture a live webcam frame to classify 7 facial expressions with instant confidence scores.",
        "cam": "Live Camera",
        "upload": "Upload Photo",
    },
    "हिन्दी": {
        "nav_scan": "स्कैनर",
        "nav_stats": "एनालिटिक्स",
        "nav_about": "हमारे बारे में",
        "nav_faq": "सवाल-जवाब",
        "login": "लॉगिन",
        "signup": "नया अकाउंट",
        "logout": "लॉग आउट",
        "pill": "एआई फेशियल एक्सप्रेशन स्कैनर",
        "hero_h1": "हर चेहरे के भाव तुरंत समझें",
        "hero_p": "फोटो अपलोड करें या लाइव वेबकैम से 7 इमोशन और कॉन्फिडेंस स्कोर तुरंत देखें।",
        "cam": "लाइव कैमरा",
        "upload": "फोटो अपलोड",
    },
    "Español": {
        "nav_scan": "Escáner",
        "nav_stats": "Analítica",
        "nav_about": "Nosotros",
        "nav_faq": "FAQ",
        "login": "Entrar",
        "signup": "Crear Cuenta",
        "logout": "Salir",
        "pill": "ESCÁNER FACIAL CON IA",
        "hero_h1": "Comprende las emociones en cada rostro",
        "hero_p": "Sube una foto o usa tu cámara para clasificar 7 expresiones faciales al instante.",
        "cam": "Cámara",
        "upload": "Subir Foto",
    },
    "Français": {
        "nav_scan": "Scanner",
        "nav_stats": "Analytique",
        "nav_about": "À Propos",
        "nav_faq": "FAQ",
        "login": "Connexion",
        "signup": "S'inscrire",
        "logout": "Déconnexion",
        "pill": "SCANNER FACIAL IA",
        "hero_h1": "Décodez les émotions sur chaque visage",
        "hero_p": "Importez une photo ou utilisez votre caméra pour analyser 7 expressions faciales.",
        "cam": "Caméra",
        "upload": "Importer Photo",
    },
}

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
defaults = {
    "page": "scan",
    "lang": "English",
    "dark_mode": False,
    "users": {"demo@affectivision.ai": {"name": "Demo User", "password": "demo123"}},
    "user": None,
    "history": [],
    "strictness": 4,
    "fallback_full": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

T = TRANSLATIONS[st.session_state.lang]
is_dark = st.session_state.dark_mode


def nav(page_key: str):
    st.session_state.page = page_key
    st.rerun()


# -----------------------------------------------------------------------------
# Design System CSS (Strict Light/Dark Contrast + Flush Top Alignment)
# -----------------------------------------------------------------------------
mode_tokens = (
    """
    --bg-page: var(--color-neutral-30);
    --bg-nav: rgba(30, 16, 64, 0.85);
    --bg-card: var(--color-neutral-15);
    --bg-soft: var(--color-accent-2);
    --border-subtle: rgba(203, 185, 255, 0.16);
    --ink-primary: #ffffff;
    --ink-secondary: var(--color-primary-4);
    --ink-muted: var(--color-neutral-8);
    --shadow-card: var(--shadow-8);
    """
    if is_dark
    else """
    --bg-page: var(--color-neutral-3);
    --bg-nav: rgba(255, 255, 255, 0.92);
    --bg-card: #ffffff;
    --bg-soft: var(--color-neutral-10);
    --border-subtle: var(--color-primary-8);
    --ink-primary: #111827;
    --ink-secondary: var(--color-neutral-2);
    --ink-muted: var(--color-neutral-33);
    --shadow-card: var(--shadow-5);
    """
)

st.markdown(
    f"""
    <style>
    :root {{
      --color-primary-1: #1e1040;
      --color-primary-2: #6c4ff6;
      --color-primary-3: #4c31a0;
      --color-primary-4: #cbb9ff;
      --color-primary-8: #ddd5f7;
      --color-primary-9: #ddd6ff;
      --color-primary-13: #7c3aed;
      --color-primary-20: #7f22fe;

      --color-neutral-2: #4b5563;
      --color-neutral-3: #f5f3ff;
      --color-neutral-8: #99a1af;
      --color-neutral-10: #eeedfe;
      --color-neutral-15: #1a1a2e;
      --color-neutral-30: #18162d;
      --color-neutral-33: #6b6385;
      --color-accent-2: #3b2a7a;

      --radius-5: 12px;
      --radius-6: 14px;
      --radius-7: 16px;
      --radius-8: 20px;
      --radius-10: 999px;

      --shadow-1: rgba(76, 49, 160, 0.1) 0px 4px 14px;
      --shadow-5: 0 10px 35px rgba(76, 49, 160, 0.08);
      --shadow-7: 0 8px 26px rgba(108, 79, 246, 0.38);
      --shadow-8: 0 18px 60px rgba(30, 16, 64, 0.28);

      {mode_tokens}
    }}

    section[data-testid="stSidebar"],
    button[data-testid="collapsedControl"],
    #MainMenu,
    header[data-testid="stHeader"],
    footer {{
      display: none !important;
    }}

    .stApp {{
      background: var(--bg-page) !important;
      color: var(--ink-primary) !important;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    /* Global text contrast for Light & Dark modes */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp p, .stApp span, .stApp label, .stApp li,
    div[data-testid="stMarkdownContainer"] *,
    div[data-testid="stWidgetLabel"] *,
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p,
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] p,
    div[data-testid="stSlider"] *,
    div[data-testid="stCheckbox"] *,
    div[data-testid="stCameraInput"] label,
    div[data-testid="stCameraInput"] p,
    div[data-testid="stCameraInput"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p,
    button[data-baseweb="tab"] {{
      color: var(--ink-primary) !important;
    }}

    /* Language Selectbox & Dropdown Menu — Strict Light/Dark Text Fix */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[role="listbox"],
    ul[data-testid="stSelectboxVirtualDropdown"],
    input, textarea {{
      background-color: var(--bg-card) !important;
      color: var(--ink-primary) !important;
      border-color: var(--border-subtle) !important;
    }}

    div[data-baseweb="select"] *,
    div[data-baseweb="popover"] *,
    div[role="listbox"] *,
    ul[data-testid="stSelectboxVirtualDropdown"] li,
    ul[data-testid="stSelectboxVirtualDropdown"] li * {{
      color: var(--ink-primary) !important;
    }}

    div[data-baseweb="select"] svg {{
      fill: var(--ink-primary) !important;
    }}

    div[data-testid="stFileUploader"] section {{
      background-color: var(--bg-card) !important;
      border: 1.5px dashed var(--border-subtle) !important;
      color: var(--ink-primary) !important;
    }}
    div[data-testid="stFileUploader"] small {{
      color: var(--ink-muted) !important;
    }}

    .block-container {{
      padding-top: 1.2rem;
      padding-bottom: 3rem;
      max-width: 1180px;
    }}

    /* Remove stray top spacing on images & result columns */
    div[data-testid="stImage"] {{
      margin-top: 0 !important;
      padding-top: 0 !important;
    }}
    div[data-testid="stImage"] img {{
      border-radius: var(--radius-7);
      display: block;
    }}

    .brand-mark {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding-top: 4px;
    }}
    .brand-icon {{
      width: 34px;
      height: 34px;
      border-radius: var(--radius-5);
      background: linear-gradient(135deg, var(--color-primary-2), var(--color-primary-20));
      box-shadow: var(--shadow-7);
      color: #ffffff !important;
      font-weight: 800;
      font-size: 1rem;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .brand-text {{
      font-size: 1.22rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: var(--ink-primary) !important;
    }}

    .hero-wrap {{
      text-align: center;
      padding: 2.4rem 1.5rem 2rem 1.5rem;
      margin: 0.5rem auto 1.8rem auto;
      background: linear-gradient(135deg, var(--color-primary-1) 0%, var(--color-primary-3) 55%, var(--color-primary-2) 100%);
      border-radius: var(--radius-8);
      box-shadow: var(--shadow-8);
    }}
    .hero-pill {{
      display: inline-block;
      padding: 6px 16px;
      border-radius: var(--radius-10);
      background: rgba(255, 255, 255, 0.14);
      border: 1px solid rgba(221, 214, 255, 0.35);
      color: #ffffff !important;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.09em;
      margin-bottom: 14px;
    }}
    .hero-h1 {{
      color: #ffffff !important;
      font-size: 2.35rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      margin: 0 0 10px 0;
      line-height: 1.15;
    }}
    .hero-p {{
      color: var(--color-primary-9) !important;
      font-size: 1.02rem;
      max-width: 640px;
      margin: 0 auto;
      line-height: 1.55;
    }}

    .surface-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-8);
      padding: 24px;
      box-shadow: var(--shadow-card);
      margin-bottom: 20px;
    }}
    .surface-card h3, .surface-card h4 {{
      color: var(--ink-primary) !important;
      margin-top: 0;
    }}
    .surface-card p {{
      color: var(--ink-secondary) !important;
      line-height: 1.6;
      margin-bottom: 0;
    }}

    .kpi-box {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-7);
      padding: 20px;
      text-align: center;
      box-shadow: var(--shadow-card);
    }}
    .kpi-num {{
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--color-primary-2) !important;
    }}
    .kpi-sub {{
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--ink-muted) !important;
      margin-top: 4px;
    }}

    .emo-panel {{
      border-radius: var(--radius-7);
      padding: 18px 20px;
      border: 1px solid var(--border-subtle);
      box-shadow: var(--shadow-1);
      margin-top: 0 !important;
      margin-bottom: 14px;
    }}
    .emo-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
    }}
    .emo-name {{
      font-size: 1.2rem;
      font-weight: 800;
      color: var(--ink-primary) !important;
    }}
    .emo-score {{
      padding: 4px 12px;
      border-radius: var(--radius-10);
      background: #ef4444;
      color: #ffffff !important;
      font-size: 0.84rem;
      font-weight: 700;
    }}
    .bar-item {{ margin-bottom: 9px; }}
    .bar-head {{
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--ink-secondary) !important;
      margin-bottom: 4px;
    }}
    .bar-bg {{
      width: 100%;
      height: 8px;
      background: var(--bg-soft);
      border-radius: var(--radius-10);
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: var(--radius-10);
    }}

    /* Standard Streamlit Buttons */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {{
      border-radius: var(--radius-10);
      font-weight: 600;
      font-size: 0.88rem;
      padding: 0.42rem 0.95rem;
      border: 1px solid var(--border-subtle);
      transition: all 0.16s ease;
    }}
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {{
      background: linear-gradient(135deg, var(--color-primary-2), var(--color-primary-13)) !important;
      color: #ffffff !important;
      border: none;
      box-shadow: var(--shadow-7);
    }}
    .stButton > button[kind="primary"] *,
    .stFormSubmitButton > button[kind="primary"] * {{
      color: #ffffff !important;
    }}
    .stButton > button[kind="secondary"],
    .stDownloadButton > button {{
      background: var(--bg-card) !important;
      color: var(--ink-primary) !important;
    }}
    .stButton > button[kind="secondary"] *,
    .stDownloadButton > button * {{
      color: var(--ink-primary) !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
      border-color: var(--color-primary-2);
    }}

    /* "Take Photo", "Clear photo" & "Browse files" Buttons */
    div[data-testid="stCameraInput"] button,
    button[data-testid="stCameraInputButton"],
    div[data-testid="stFileUploader"] button {{
      background: linear-gradient(135deg, var(--color-primary-2), var(--color-primary-13)) !important;
      color: #ffffff !important;
      border: none !important;
      border-radius: var(--radius-10) !important;
      font-weight: 700 !important;
      padding: 0.48rem 1.2rem !important;
      box-shadow: var(--shadow-7) !important;
    }}
    div[data-testid="stCameraInput"] button *,
    button[data-testid="stCameraInputButton"] *,
    div[data-testid="stFileUploader"] button * {{
      color: #ffffff !important;
    }}

    div[data-testid="stExpander"] {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-6);
      box-shadow: var(--shadow-1);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model & Vision Engine (Red Rectangle + Multi-Cascade Face Detection)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        return None, []
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    cascade_files = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_alt.xml",
    ]
    cascades = []
    for fname in cascade_files:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + fname)
        if not c.empty():
            cascades.append(c)
    return model, cascades


def _has_rescaling_layer(model) -> bool:
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True
    return False


def _prepare_face_tensor(rgb_roi: np.ndarray, model) -> np.ndarray:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    target_h = shape[1] if len(shape) >= 3 and shape[1] else 48
    target_w = shape[2] if len(shape) >= 3 and shape[2] else 48
    channels = shape[3] if len(shape) == 4 and shape[3] else 1

    if channels == 1:
        gray_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray_roi, (target_w, target_h), interpolation=cv2.INTER_AREA)
        arr = resized.astype("float32")
        if not _has_rescaling_layer(model):
            arr = arr / 255.0
        return arr.reshape(1, target_h, target_w, 1)
    else:
        resized = cv2.resize(rgb_roi, (target_w, target_h), interpolation=cv2.INTER_AREA)
        arr = resized.astype("float32")
        if not _has_rescaling_layer(model):
            arr = arr / 255.0
        return arr.reshape(1, target_h, target_w, channels)


def detect_emotions(pil_img, model, cascades, min_neighbors=4, fallback_full=True):
    # 1. Fix phone/camera EXIF rotation
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")

    # 2. Resize huge photos for faster, more accurate detection
    max_side = 1000
    if max(pil_img.size) > max_side:
        pil_img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    rgb = np.array(pil_img)
    h_img, w_img = rgb.shape[:2]

    # 3. Pad image by 12% so close-up webcam selfies don't clip the face border
    pad_y, pad_x = int(h_img * 0.12), int(w_img * 0.12)
    padded_rgb = cv2.copyMakeBorder(
        rgb, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
    )
    gray = cv2.cvtColor(padded_rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # 4. Multi-pass detection across cascades
    faces = []
    for cascade in cascades:
        for img_pass in (gray_eq, gray):
            found = cascade.detectMultiScale(
                img_pass,
                scaleFactor=1.1,
                minNeighbors=min_neighbors,
                minSize=(28, 28),
            )
            if len(found) > 0:
                faces = found
                break
        if len(faces) > 0:
            break

    # Convert padded coordinates back to original image coordinates
    boxes = []
    for (x, y, w, h) in faces:
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w - pad_x)
        y2 = min(h_img, y + h - pad_y)
        if (x2 - x1) > 15 and (y2 - y1) > 15:
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    used_fallback = False
    if not boxes and fallback_full:
        margin = 6
        boxes = [(margin, margin, max(1, w_img - 2 * margin), max(1, h_img - 2 * margin))]
        used_fallback = True

    annotated = rgb.copy()
    detections = []

    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        roi_rgb = rgb[y : y + h, x : x + w]
        inp = _prepare_face_tensor(roi_rgb, model)

        raw_preds = model.predict(inp, verbose=0)[0]
        if np.min(raw_preds) < 0 or not np.isclose(np.sum(raw_preds), 1.0, atol=1e-2):
            probs = tf.nn.softmax(raw_preds).numpy()
        else:
            probs = raw_preds

        num_classes = min(len(probs), len(EMOTION_LABELS))
        top_i = int(np.argmax(probs[:num_classes]))
        label = EMOTION_LABELS[top_i]
        conf = float(probs[top_i]) * 100.0

        # Draw RED rectangle around the face
        cv2.rectangle(annotated, (x, y), (x + w, y + h), RED_BOX_RGB, 3)

        # Draw red label banner with emotion + confidence percentage
        tag = f"{label}: {conf:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.55, min(w / 260.0, 0.9))
        thickness = 2
        (tw, th), _ = cv2.getTextSize(tag, font, font_scale, thickness)

        tag_y1 = max(y - th - 14, 0)
        tag_y2 = tag_y1 + th + 14
        cv2.rectangle(annotated, (x, tag_y1), (x + tw + 14, tag_y2), RED_BOX_RGB, -1)
        cv2.putText(
            annotated,
            tag,
            (x + 7, tag_y2 - 7),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        detections.append({
            "id": idx,
            "emotion": label,
            "confidence": conf,
            "fallback": used_fallback,
            "probs": {EMOTION_LABELS[i]: float(probs[i]) * 100.0 for i in range(num_classes)},
        })

    return annotated, detections


def to_png_bytes(rgb_arr):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else b""


# -----------------------------------------------------------------------------
# Top Navigation Bar
# -----------------------------------------------------------------------------
n_brand, n_scan, n_stats, n_about, n_faq, n_lang, n_theme, n_auth = st.columns(
    [2.3, 1.0, 1.05, 0.9, 0.85, 1.3, 0.8, 1.3],
    vertical_alignment="center",
)

with n_brand:
    st.markdown(
        """
        <div class="brand-mark">
            <div class="brand-icon">A</div>
            <div class="brand-text">AffectiVision</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with n_scan:
    if st.button(T["nav_scan"], type="primary" if st.session_state.page == "scan" else "secondary", use_container_width=True):
        nav("scan")

with n_stats:
    if st.button(T["nav_stats"], type="primary" if st.session_state.page == "stats" else "secondary", use_container_width=True):
        nav("stats")

with n_about:
    if st.button(T["nav_about"], type="primary" if st.session_state.page == "about" else "secondary", use_container_width=True):
        nav("about")

with n_faq:
    if st.button(T["nav_faq"], type="primary" if st.session_state.page == "faq" else "secondary", use_container_width=True):
        nav("faq")

with n_lang:
    sel_lang = st.selectbox(
        "Lang",
        list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.lang),
        label_visibility="collapsed",
    )
    if sel_lang != st.session_state.lang:
        st.session_state.lang = sel_lang
        st.rerun()

with n_theme:
    theme_icon = "☀️" if is_dark else "🌙"
    if st.button(theme_icon, help="Toggle Light / Dark theme", use_container_width=True):
        st.session_state.dark_mode = not is_dark
        st.rerun()

with n_auth:
    if st.session_state.user:
        uname = st.session_state.users[st.session_state.user]["name"].split()[0]
        if st.button(f"👤 {uname}", use_container_width=True):
            nav("auth")
    else:
        if st.button(T["login"], type="primary", use_container_width=True):
            nav("auth")

st.divider()

model, face_cascades = load_resources()

# =============================================================================
# PAGE 1: SCANNER
# =============================================================================
if st.session_state.page == "scan":
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-pill">{T['pill']}</div>
            <h1 class="hero-h1">{T['hero_h1']}</h1>
            <p class="hero-p">{T['hero_p']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(f"Model file not found at `{MODEL_PATH}`. Run `train_and_evaluate.py` first.")
        st.stop()

    c_mode, c_strict, c_fb = st.columns([1.6, 1.2, 1.2], vertical_alignment="center")
    with c_mode:
        input_mode = st.radio("Source", [T["upload"], T["cam"]], horizontal=True, label_visibility="collapsed")
    with c_strict:
        st.session_state.strictness = st.slider("Detection Strictness", 2, 8, st.session_state.strictness)
    with c_fb:
        st.session_state.fallback_full = st.checkbox(
            "Analyze full image if no face box found",
            value=st.session_state.fallback_full,
        )

    img_input = None
    if input_mode == T["cam"]:
        cam_shot = st.camera_input("Capture a photo")
        if cam_shot is not None:
            img_input = Image.open(cam_shot)
    else:
        up_file = st.file_uploader("Drop a face image here (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if up_file is not None:
            img_input = Image.open(up_file)

    if img_input is not None:
        with st.spinner("Scanning facial expressions..."):
            annotated, detections = detect_emotions(
                img_input,
                model,
                face_cascades,
                st.session_state.strictness,
                st.session_state.fallback_full,
            )

        col_img, col_out = st.columns([1.25, 1], gap="large", vertical_alignment="top")
        with col_img:
            st.image(annotated, use_container_width=True)
            st.download_button(
                "Download Result PNG",
                data=to_png_bytes(annotated),
                file_name=f"affectivision_{datetime.now().strftime('%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
            )

        with col_out:
            if not detections:
                st.warning("No face detected. Lower Detection Strictness or enable full-image analysis.")
            else:
                if detections[0].get("fallback"):
                    st.info("No face box found — analyzed the full image directly.")
                for d in detections:
                    st.session_state.history.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Emotion": d["emotion"],
                        "Confidence (%)": round(d["confidence"], 1),
                    })
                    pal = EMOTION_PALETTE[d["emotion"]]
                    panel_bg = pal["dark"] if is_dark else pal["light"]

                    bars = ""
                    for emo, val in sorted(d["probs"].items(), key=lambda x: x[1], reverse=True):
                        b_hex = EMOTION_PALETTE[emo]["hex"]
                        bars += f"""
                        <div class="bar-item">
                            <div class="bar-head"><span>{emo}</span><span>{val:.1f}%</span></div>
                            <div class="bar-bg"><div class="bar-fill" style="width:{min(val,100):.1f}%; background:{b_hex};"></div></div>
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div class="emo-panel" style="background:{panel_bg};">
                            <div class="emo-top">
                                <div class="emo-name">Face #{d['id']} · {d['emotion']}</div>
                                <div class="emo-score">Confidence: {d['confidence']:.1f}%</div>
                            </div>
                            {bars}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# =============================================================================
# PAGE 2: ANALYTICS
# =============================================================================
elif st.session_state.page == "stats":
    st.subheader("Session Analytics")
    if not st.session_state.history:
        st.info("Scan a face in **Scanner** first to populate live session analytics.")
    else:
        df = pd.DataFrame(st.session_state.history)
        k1, k2, k3 = st.columns(3)
        k1.markdown(f'<div class="kpi-box"><div class="kpi-num">{len(df)}</div><div class="kpi-sub">Total Faces</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi-box"><div class="kpi-num">{df["Emotion"].mode()[0]}</div><div class="kpi-sub">Top Expression</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi-box"><div class="kpi-num">{df["Confidence (%)"].mean():.1f}%</div><div class="kpi-sub">Avg Confidence</div></div>', unsafe_allow_html=True)

        st.markdown("#### Expression Breakdown")
        st.bar_chart(df["Emotion"].value_counts(), color="#6c4ff6")
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Export CSV", df.to_csv(index=False).encode("utf-8"), "affectivision_log.csv", "text/csv", use_container_width=True)
        with c2:
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

# =============================================================================
# PAGE 3: ABOUT
# =============================================================================
elif st.session_state.page == "about":
    st.subheader("About AffectiVision")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="surface-card"><h4>🔒 100% Local Processing</h4><p>Images are analyzed in memory on your device and never uploaded to third-party servers.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="surface-card"><h4>⚡ 7-Class Deep CNN</h4><p>Detects Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise from facial features.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="surface-card"><h4>📊 Instant Export</h4><p>Download annotated PNG frames and CSV analytics logs directly from your browser.</p></div>', unsafe_allow_html=True)

# =============================================================================
# PAGE 4: FAQs
# =============================================================================
elif st.session_state.page == "faq":
    st.subheader("Frequently Asked Questions")
    with st.expander("What emotions does AffectiVision detect?"):
        st.write("Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.")
    with st.expander("Why did it miss a face in my photo?"):
        st.write("Make sure the face is well-lit and facing forward, lower the **Detection Strictness** slider, or keep **Analyze full image if no face box found** enabled.")
    with st.expander("Are my photos saved anywhere?"):
        st.write("No. Everything stays in temporary browser session memory and clears when you close the tab.")
    with st.expander("What is the demo login?"):
        st.write("Use `demo@affectivision.ai` and password `demo123`, or create your own account.")

# =============================================================================
# PAGE 5: AUTH (LOGIN / SIGN UP)
# =============================================================================
elif st.session_state.page == "auth":
    _, center_col, _ = st.columns([1, 1.4, 1])
    with center_col:
        if st.session_state.user:
            info = st.session_state.users[st.session_state.user]
            st.success(f"Signed in as **{info['name']}** (`{st.session_state.user}`)")
            if st.button(T["logout"], type="primary", use_container_width=True):
                st.session_state.user = None
                nav("scan")
        else:
            t_in, t_up = st.tabs([T["login"], T["signup"]])
            with t_in:
                with st.form("signin_form"):
                    em = st.text_input("Email", value="demo@affectivision.ai")
                    pw = st.text_input("Password", type="password", value="demo123")
                    if st.form_submit_button(T["login"], type="primary", use_container_width=True):
                        u = st.session_state.users.get(em.strip().lower())
                        if u and u["password"] == pw:
                            st.session_state.user = em.strip().lower()
                            nav("scan")
                        else:
                            st.error("Invalid email or password.")
            with t_up:
                with st.form("signup_form"):
                    nm = st.text_input("Full Name", placeholder="Alex Rivera")
                    em = st.text_input("Email", placeholder="alex@example.com")
                    pw = st.text_input("Password", type="password")
                    if st.form_submit_button(T["signup"], type="primary", use_container_width=True):
                        clean = em.strip().lower()
                        if not nm or not clean or len(pw) < 4:
                            st.warning("Fill in all fields (password min 4 characters).")
                        elif clean in st.session_state.users:
                            st.error("Account already exists.")
                        else:
                            st.session_state.users[clean] = {"name": nm.strip(), "password": pw}
                            st.session_state.user = clean
                            nav("scan")