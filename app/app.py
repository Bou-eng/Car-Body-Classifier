from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import (  # noqa: E402
    CLASS_TO_NUMBER,
    MODEL_PATH,
    PROJECT_CLASS_ORDER,
    load_model,
    predict_loaded_image,
)


st.set_page_config(
    page_title="Araba Gövde Tipi Sınıflandırma",
    page_icon="🚗",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #e7f1ed;
        --muted: #a5bbb4;
        --line: #2d4842;
        --paper: #0d1b19;
        --panel: #142825;
        --accent: #4ec5b4;
        --accent-dark: #b4f0e4;
        --accent-soft: #183b36;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp small,
    [data-testid='stMarkdownContainer'] {
        color: var(--ink);
    }
    [data-testid='stCaptionContainer'], [data-testid='stCaptionContainer'] p {
        color: var(--muted) !important;
    }

    [data-testid='stHeader'], [data-testid='stMain'] { background: var(--paper); }
    [data-testid='stSidebar'] {
        background: #102421;
        border-right: 0;
    }
    [data-testid='stSidebar'] * { color: #e7f1ed; }
    [data-testid='stSidebar'] hr { border-color: rgba(231, 241, 237, .16); }
    [data-testid='stSidebar'] [data-testid='stMetric'] {
        background: rgba(255, 255, 255, .08);
        border-color: rgba(231, 241, 237, .14);
    }
    [data-testid='stSidebar'] [data-testid='stMetricValue'] { color: #ffffff !important; }
    [data-testid='stSidebar'] [data-testid='stMetricLabel'] { color: #a9c2ba !important; }
    [data-testid='stSidebar'] [data-testid='stCaptionContainer'],
    [data-testid='stSidebar'] [data-testid='stCaptionContainer'] p { color: #a9c2ba !important; }

    .block-container {
        max-width: 1400px;
        padding: 2.5rem 4rem 4rem;
    }

    h1, h2, h3, [data-testid='stMetricValue'] {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0;
    }
    h1 { color: var(--ink); font-size: clamp(2rem, 4vw, 3.4rem); line-height: 1.05; }
    h2, h3 { color: var(--ink); }
    p, label, [data-testid='stCaptionContainer'] { color: var(--muted); }

    .topline {
        align-items: center;
        display: flex;
        gap: .65rem;
        margin-bottom: 1.4rem;
    }
    .brand-mark {
        align-items: center;
        background: var(--accent);
        border-radius: 12px;
        color: #fff !important;
        display: inline-flex;
        font-size: 1.25rem;
        height: 42px;
        justify-content: center;
        width: 42px;
    }
    .eyebrow {
        color: var(--accent);
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero-copy { max-width: 680px; }
    .hero-copy p { font-size: 1.05rem; line-height: 1.65; margin-top: -.5rem; }

    .workspace {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 14px 40px rgba(25, 55, 47, .07);
        padding: 1.35rem;
    }
    .panel-label {
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: .7rem;
    }
    .panel-note { color: var(--muted); font-size: .88rem; margin-top: -.45rem; }
    .empty-state {
        align-items: center;
        background: #102420;
        border: 1px dashed #4b8176;
        border-radius: 14px;
        color: var(--muted);
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 280px;
        padding: 2rem;
        text-align: center;
    }
    .empty-icon { font-size: 2.2rem; margin-bottom: .8rem; }
    .result-hero {
        background: var(--accent-soft);
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        margin: .25rem 0 1rem;
        padding: 1.15rem 1.25rem;
    }
    .result-kicker { color: var(--accent-dark); font-size: .78rem; font-weight: 700; text-transform: uppercase; }
    .result-name { color: var(--accent-dark); font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; }
    .result-subtitle { color: #9ed8cc; font-size: .9rem; }
    .section-rule { border-top: 1px solid var(--line); margin: 1.5rem 0; }
    .sidebar-title { color: #fff; font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600; }
    .sidebar-kicker { color: #8fb6aa; font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .class-chip {
        background: rgba(226, 241, 237, .09);
        border: 1px solid rgba(226, 241, 237, .12);
        border-radius: 8px;
        color: #e7f1ed !important;
        display: inline-block;
        font-size: .78rem;
        margin: .18rem .12rem;
        padding: .32rem .5rem;
    }
    [data-testid='stFileUploaderDropzone'] {
        background: #102420;
        border: 1px dashed #4b8176;
        border-radius: 12px;
    }
    [data-testid='stFileUploaderDropzone'] small,
    [data-testid='stFileUploaderDropzone'] span { color: var(--muted) !important; }
    .stButton > button {
        background: var(--accent);
        border: 0;
        border-radius: 9px;
        color: #fff;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        min-height: 2.8rem;
        width: 100%;
    }
    .stButton > button:hover { background: var(--accent-dark); color: #fff; }
    [data-testid='stMetric'] {
        background: rgba(255, 255, 255, .08);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: .8rem;
    }
    [data-testid='stMetricLabel'] { color: var(--muted); }
    [data-testid='stMetricValue'] { color: var(--ink); font-size: 1.3rem; }
    .empty-state, .empty-state strong { color: var(--muted) !important; }
    [data-testid='stDataFrame'] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    [data-testid='stDataFrame'] * { color: var(--ink) !important; }

    @media (max-width: 800px) {
        .block-container { padding: 1.5rem 1rem 3rem; }
        .workspace { padding: 1rem; }
        .result-name { font-size: 1.65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_cached_model():
    return load_model(MODEL_PATH)


def predict_uploaded_image(image: Image.Image):
    model, classes, device, checkpoint = load_cached_model()
    result = predict_loaded_image(
        image=image,
        model=model,
        classes=classes,
        device=device,
        checkpoint=checkpoint,
    )
    result["device"] = str(device)
    return result


if not MODEL_PATH.exists():
    st.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
    st.stop()

model, classes, device, checkpoint = load_cached_model()

with st.sidebar:
    st.markdown('<div class="sidebar-kicker">Inference console</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Model bilgisi</div>', unsafe_allow_html=True)
    st.caption("Canlı tahmin motoru")
    st.metric("Model", checkpoint.get("model_name", "Bilinmiyor").replace("_", " ").title())
    st.metric("Cihaz", str(device).upper())

    if "best_val_f1" in checkpoint:
        st.metric("Best validation F1", f"{checkpoint['best_val_f1']:.4f}")

    st.divider()
    image_size = checkpoint.get("image_size", 224)
    st.caption(f"Girdi: {image_size} x {image_size}px  |  {len(classes)} sınıf")
    st.markdown('<div class="sidebar-title">Sınıf kataloğu</div>', unsafe_allow_html=True)
    class_chips = "".join(
        f'<span class="class-chip">{CLASS_TO_NUMBER[class_name]} · {class_name}</span>'
        for class_name in PROJECT_CLASS_ORDER
    )
    st.markdown(class_chips, unsafe_allow_html=True)

st.markdown(
    """
    <div class="topline">
        <span class="brand-mark">🚗</span>
        <span class="eyebrow">Automotive vision system</span>
    </div>
    <div class="hero-copy">
        <h1>Car body type<br>classification</h1>
        <p>Yüklediğiniz görseli analiz edin ve eğitilmiş modelin en güçlü tahminini saniyeler içinde görün.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

upload_col, status_col = st.columns([1.55, 1], gap="large")

with upload_col:
    st.markdown('<div class="panel-label">01 · Görsel seçin</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">JPG, PNG, WEBP veya BMP formatında bir araç fotoğrafı yükleyin.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Bir araba görseli yükle",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

with status_col:
    st.markdown('<div class="panel-label">02 · Sistem durumu</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-hero"><div class="result-kicker">Ready to analyze</div>'
        f'<div class="result-name">{checkpoint.get("model_name", "Model").replace("_", " ").title()}</div>'
        f'<div class="result-subtitle">Model loaded · {str(device).upper()} · {image_size}x{image_size}</div></div>',
        unsafe_allow_html=True,
    )

if uploaded_file is None:
    st.markdown(
        '<div class="empty-state"><div class="empty-icon">◌</div>'
        '<strong>Analiz için bir görsel bekleniyor</strong>'
        '<span>Sonuçlar burada görünecek.</span></div>',
        unsafe_allow_html=True,
    )
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
upload_id = f"{uploaded_file.name}:{uploaded_file.size}"
if st.session_state.get("upload_id") != upload_id:
    st.session_state["upload_id"] = upload_id
    st.session_state.pop("prediction_result", None)

preview_col, result_col = st.columns([1, 1], gap="large")

with preview_col:
    st.markdown('<div class="panel-label">Görsel önizleme</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.caption(f"{uploaded_file.name} · {image.width} x {image.height}px")

with result_col:
    st.markdown('<div class="panel-label">Tahmin sonucu</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">Analizi başlatmak için aşağıdaki butona tıklayın.</div>', unsafe_allow_html=True)
    if st.button("Görseli analiz et", type="primary"):
        result = predict_uploaded_image(image)
        st.session_state["prediction_result"] = result

    result = st.session_state.get("prediction_result")
    if result is not None:
        st.markdown(
            f'<div class="result-hero"><div class="result-kicker">Predicted body type</div>'
            f'<div class="result-name">{result["predicted_class"]}</div>'
            f'<div class="result-subtitle">Sınıf {result["predicted_number"]} · {result["model_name"]}</div></div>',
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Güven", f"%{result['confidence'] * 100:.1f}")
        with metric_col2:
            st.metric("Sınıf", result["predicted_number"])
        with metric_col3:
            st.metric("Süre", f"{result['elapsed_time']:.3f}s")


if result is not None:
    probabilities_df = pd.DataFrame(
        {
            "Sınıf": list(result["probabilities"].keys()),
            "Olasılık": [value * 100 for value in result["probabilities"].values()],
        }
    ).sort_values("Olasılık", ascending=False)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Güven dağılımı</div>', unsafe_allow_html=True)
    chart_col, table_col = st.columns([1.1, .9], gap="large")
    with chart_col:
        st.bar_chart(
            probabilities_df,
            x="Sınıf",
            y="Olasılık",
            use_container_width=True,
        )
    with table_col:
        st.dataframe(
            probabilities_df,
            use_container_width=True,
            hide_index=True,
            column_config={"Olasılık": st.column_config.ProgressColumn("Olasılık", format="%.1f%%", min_value=0, max_value=100)},
        )