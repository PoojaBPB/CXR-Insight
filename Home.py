import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps
import base64


# Page setup

st.set_page_config(
    page_title="CXR Insight",
    page_icon="🫁",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"


# Global style

st.markdown("""
<style>

.block-container {
    padding-top: 0rem !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
    padding-bottom: 3rem !important;
}

[data-testid="stSidebarHeader"] {
    height: 285px !important;
    min-height: 285px !important;
}

[data-testid="stSidebarHeader"] img {
    width: 250px !important;
    height: 250px !important;
    max-height: none !important;
    object-fit: contain;
}

[data-testid="stSidebarNav"] {
    margin-top: 15px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
}

[data-testid="stPageLink"] {
    border-radius: 8px;
}

p {
    line-height: 1.65;
}

.feature-text {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 15px;
    line-height: 1.7;
    text-align: justify;
    color: var(--text-color);
    opacity: 0.86;
    min-height: 78px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# Sidebar logo

st.logo(
    str(IMAGES_DIR / "2.png"),
    size="large"
)


# Hero banner

banner = base64.b64encode(
    (IMAGES_DIR / "1.png").read_bytes()
).decode()

st.html(f"""
<div style="
    width:100%;
    height:260px;
    position:relative;
    overflow:hidden;
    background:#021220;

    background-image:
        linear-gradient(
            90deg,
            rgba(2,18,32,0.94),
            rgba(2,18,32,0.55),
            rgba(2,18,32,0.05)
        ),
        url('data:image/png;base64,{banner}');

    background-size:cover, contain;
    background-position:center, right center;
    background-repeat:no-repeat;
">

    <div style="
        position:absolute;
        left:65px;
        top:50%;
        transform:translateY(-50%);
    ">

        <div style="
            font-family:'Segoe UI', Arial, sans-serif;
            font-size:66px;
            font-weight:700;
            letter-spacing:-2px;
            color:white;
        ">
            CXR Insight
        </div>

        <div style="
            font-family:Georgia, serif;
            font-size:18px;
            font-style:italic;
            color:#d8e8f5;
            margin-top:8px;
        ">
            Bringing AI to TB Screening
        </div>

    </div>
</div>
""")


# About CXR Insight

st.html("""
<div style="
    max-width:1050px;
    margin:28px auto 18px auto;
    padding:27px 31px;

    border:1px solid rgba(120,150,170,0.24);
    border-top:3px solid #2FA7E0;
    border-radius:12px;

    background:var(--secondary-background-color);
    box-shadow:0 5px 18px rgba(0,0,0,0.035);
">

    <div style="
        font-family:'Segoe UI', Arial, sans-serif;
        font-size:27px;
        font-weight:650;
        color:var(--text-color);
        margin-bottom:12px;
    ">
        About CXR Insight
    </div>

    <div style="
        font-family:'Segoe UI', Arial, sans-serif;
        font-size:15px;
        line-height:1.75;
        text-align:justify;
        color:var(--text-color);
        opacity:0.86;
    ">
        <strong>CXR Insight</strong> offers an interactive way to explore how
        artificial intelligence interprets chest X-rays in the context of
        tuberculosis screening. Each prediction is accompanied by a visual
        explanation that shows which areas of the image influenced the model's
        decision, so you can see not only what it concluded but why. Browse a
        set of sample cases to see how the model performs against known
        outcomes, or upload your own chest X-ray to experience the tool firsthand.
    </div>

</div>
""")


# Using CXR Insight

st.html("""
<div style="
    text-align:center;
    margin:22px auto 18px auto;
">

    <div style="
        font-family:'Segoe UI', Arial, sans-serif;
        font-size:28px;
        font-weight:650;
        color:var(--text-color);
    ">
        Using CXR Insight
    </div>

    <div style="
        width:54px;
        height:3px;
        background:#2FA7E0;
        margin:8px auto 0 auto;
        border-radius:4px;
    "></div>

</div>
""")


# Feature images

case_image = Image.open(
    IMAGES_DIR / "3.png"
).convert("RGB")

upload_image = Image.open(
    IMAGES_DIR / "4.png"
).convert("RGB")


case_image = ImageOps.pad(
    case_image,
    (400, 145),
    method=Image.Resampling.LANCZOS,
    color=(245, 249, 252)
)

upload_image = ImageOps.pad(
    upload_image,
    (400, 145),
    method=Image.Resampling.LANCZOS,
    color=(245, 249, 252)
)


# Feature cards

space_left, case_col, upload_col, space_right = st.columns(
    [1.25, 2, 2, 1.25],
    gap="large"
)


# Case Explorer

with case_col:

    with st.container(border=True):

        st.image(
            case_image,
            use_container_width=True
        )

        st.markdown("### 🗂️ Case Explorer")

        st.markdown(
            """
            <div class="feature-text">
                Review chest X-rays with known outcomes, compare the model
                prediction with the confirmed result, and explore the visual
                explanation.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.page_link(
            "pages/1_Case_Explorer.py",
            label="Open Case Explorer →",
            use_container_width=True
        )


# Upload & Analyse

with upload_col:

    with st.container(border=True):

        st.image(
            upload_image,
            use_container_width=True
        )

        st.markdown("### 📤 Upload & Analyse")

        st.markdown(
            """
            <div class="feature-text">
                Upload your own chest X-ray for analysis, review the model
                prediction and TB Prediction Score, and explore the visual
                explanation.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.page_link(
            "pages/2_Upload_and_Analyse.py",
            label="Start Analysis →",
            use_container_width=True
        )


# Model Behind CXR Insight

st.html("""
<div style="
    max-width:1050px;
    margin:28px auto 0 auto;
    padding:27px 31px;

    border:1px solid rgba(120,150,170,0.24);
    border-top:3px solid #2FA7E0;
    border-radius:12px;

    background:var(--secondary-background-color);
    box-shadow:0 5px 18px rgba(0,0,0,0.035);
">

    <div style="
        font-family:'Segoe UI', Arial, sans-serif;
        font-size:27px;
        font-weight:650;
        color:var(--text-color);
        margin-bottom:12px;
    ">
        Model Behind CXR Insight
    </div>

    <div style="
        font-family:'Segoe UI', Arial, sans-serif;
        font-size:15px;
        line-height:1.75;
        text-align:justify;
        color:var(--text-color);
        opacity:0.86;
    ">
        CXR Insight is powered by a fine-tuned ResNet50 deep-learning model
        developed using the TBX11K chest X-ray dataset to distinguish between
        TB and non-TB cases. The model achieved an AUROC of
        0.9873 during internal validation, showing strong
        ability to separate the two classes. Predictions can also be supported
        by Grad-CAM++ visualisations, providing a simple visual indication of
        the image regions that contributed to the model's output.
    </div>

</div>
""")


# Disclaimer

st.html("""
<div style="
    max-width:1050px;
    margin:18px auto 36px auto;
    padding:20px 25px;

    border:1px solid rgba(205,158,72,0.30);
    border-left:4px solid #D5A13E;
    border-radius:10px;

    background:rgba(205,158,72,0.06);
">

    <div style="
        font-family:'Segoe UI', Arial, sans-serif;
        font-size:18px;
        font-weight:650;
        color:var(--text-color);
        margin-bottom:7px;
    ">
        Disclaimer
    </div>

    <div style="
        font-family:'Segoe UI', Arial, sans-serif;
        font-size:14.5px;
        line-height:1.7;
        color:var(--text-color);
        opacity:0.85;
    ">
        CXR Insight is a research and educational demonstration only, not a
        clinical diagnostic tool, and its predictions should not be used to
        inform real medical decisions.
    </div>

</div>
""")