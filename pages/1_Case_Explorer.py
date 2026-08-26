import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import base64
import numpy as np
import tensorflow as tf
import matplotlib
st.set_page_config(page_title='Case Explorer | CXR Insight', page_icon='🫁', layout='wide')
BASE_DIR = Path(__file__).resolve().parents[1]
CASE_IMAGES_DIR = BASE_DIR / 'case_explorer' / 'case_images'
ANNOTATIONS_DIR = BASE_DIR / 'case_explorer' / 'annotations'
IMAGES_DIR = BASE_DIR / 'images'
MODEL_PATH = BASE_DIR / 'models' / 'best_resnet50_finetuned.keras'
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.698488
st.markdown("""
<style>

.block-container {
    max-width: 1180px;
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
}

p {
    line-height: 1.65;
}


/* Sidebar */

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


/* Header */

.case-title {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 45px;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--text-color);
    margin-top: 14px;
    margin-bottom: 15px;
}

.intro-box {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 15px;
    line-height: 1.75;
    text-align: justify;
    color: var(--text-color);

    padding: 19px 23px;

    background: rgba(47, 167, 224, 0.055);

    border: 1px solid rgba(47, 167, 224, 0.25);
    border-left: 4px solid #2FA7E0;
    border-radius: 12px;
}


/* Section headings */

.section-title {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 27px;
    font-weight: 650;
    color: var(--text-color);
    margin-bottom: 5px;
}

.section-subtitle {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14.5px;
    line-height: 1.65;
    color: var(--text-color);
    opacity: 0.70;
    margin-bottom: 15px;
}

.case-heading {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 19px;
    font-weight: 650;
    text-align: center;
    color: var(--text-color);
    margin-bottom: 11px;
}


/* Result cards */

.result-card {
    width: 100%;
    height: 160px;
    box-sizing: border-box;

    padding: 20px;

    border: 1px solid rgba(120, 150, 170, 0.25);
    border-top: 3px solid #2FA7E0;
    border-radius: 13px;

    background: var(--secondary-background-color);

    display: flex;
    flex-direction: column;
    justify-content: center;

    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.025);
}

.result-label {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.75px;
    text-transform: uppercase;
    color: var(--text-color);
    opacity: 0.58;
    margin-bottom: 11px;
}

.result-value {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 25px;
    font-weight: 700;
    line-height: 1.25;
    color: var(--text-color);
    margin-bottom: 8px;
}

.result-note {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    color: var(--text-color);
    opacity: 0.62;
}


/* Prediction status */

.correct-box {
    margin-top: 17px;
    padding: 14px 18px;

    border: 1px solid rgba(58, 175, 115, 0.30);
    border-left: 4px solid #3AAF73;
    border-radius: 10px;

    background: rgba(58, 175, 115, 0.07);

    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-color);
}

.incorrect-box {
    margin-top: 17px;
    padding: 14px 18px;

    border: 1px solid rgba(215, 116, 87, 0.32);
    border-left: 4px solid #D77457;
    border-radius: 10px;

    background: rgba(215, 116, 87, 0.07);

    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-color);
}


/* Visualisation text */

.visual-info {
    margin-top: 24px;
    padding: 22px 25px;

    border: 1px solid rgba(47, 167, 224, 0.25);
    border-left: 4px solid #2FA7E0;
    border-radius: 12px;

    background: rgba(47, 167, 224, 0.055);

    color: var(--text-color);
}

.visual-info-title {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 19px;
    font-weight: 650;
    margin-bottom: 11px;
}

.visual-info-text {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14.5px;
    line-height: 1.8;
    text-align: justify;
    color: var(--text-color);
    opacity: 0.86;
}


/* Disclaimer */

.disclaimer-box {
    padding: 19px 23px;

    border: 1px solid rgba(205, 158, 72, 0.30);
    border-left: 4px solid #D5A13E;
    border-radius: 10px;

    background: rgba(205, 158, 72, 0.06);

    color: var(--text-color);
}

.disclaimer-title {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 18px;
    font-weight: 650;
    margin-bottom: 7px;
}

.disclaimer-text {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14.5px;
    line-height: 1.7;
    opacity: 0.85;
}


/* Divider */

.soft-divider {
    height: 1px;
    background: rgba(120, 150, 170, 0.18);
    margin: 31px 0;
}


/* Buttons */

div.stButton > button {
    min-height: 46px;
    border-radius: 9px;
    font-weight: 600;
}

button[kind="primary"] {
    background-color: #2FA7E0 !important;
    border-color: #2FA7E0 !important;
    color: white !important;
}

button[kind="primary"]:hover {
    background-color: #258FC0 !important;
    border-color: #258FC0 !important;
}


/* Select box */

div[data-baseweb="select"] > div {
    border-radius: 9px;
}


/* Bordered containers */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 13px !important;
}

</style>
""", unsafe_allow_html=True)
logo_path = IMAGES_DIR / '2.png'
if logo_path.exists():
    st.logo(str(logo_path), size='large')

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)
if not MODEL_PATH.exists():
    st.error(f'Model file not found: {MODEL_PATH}')
    st.stop()
try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f'Unable to load the model: {error}')
    st.stop()

@st.cache_resource
def setup_gradcam(_model):
    resnet_base = None
    for layer in _model.layers:
        if 'resnet' in layer.name.lower() and isinstance(layer, tf.keras.Model):
            resnet_base = layer
            break
    if resnet_base is None:
        raise ValueError('ResNet50 backbone could not be found.')
    last_conv_layer = resnet_base.get_layer('conv5_block3_out')
    feature_model = tf.keras.Model(inputs=resnet_base.input, outputs=last_conv_layer.output)
    gap_layer = _model.layers[-2]
    classifier = _model.layers[-1]
    return (feature_model, gap_layer, classifier)
try:
    feature_model, gap_layer, classifier = setup_gradcam(model)
except Exception as error:
    st.error(f'Grad-CAM++ could not be prepared: {error}')
    st.stop()

def get_case_images():
    if not CASE_IMAGES_DIR.exists():
        return []
    extensions = {'.png', '.jpg', '.jpeg'}
    images = [file for file in CASE_IMAGES_DIR.iterdir() if file.suffix.lower() in extensions]
    return sorted(images, key=lambda file: file.name.lower())
case_images = get_case_images()
if not case_images:
    st.error('No case images were found.')
    st.stop()

def get_known_label(image_path):
    name = image_path.stem.lower()
    if name.startswith('tb'):
        return 1
    if name.startswith('h') or name.startswith('s'):
        return 0
    raise ValueError(f'Known result could not be identified for {image_path.name}')

def get_annotation_path(image_path):
    annotation_path = ANNOTATIONS_DIR / f'{image_path.stem}.xml'
    if annotation_path.exists():
        return annotation_path
    return None

def prepare_model_input(image_path):
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image

def predict_case(image_path):
    image = prepare_model_input(image_path)
    output = model(image, training=False)
    score = float(tf.reshape(output, [-1])[0].numpy())
    prediction = int(score >= THRESHOLD)
    return (score, prediction)

def generate_gradcam_plus_plus(image_path, predicted_class):
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_output = feature_model(image, training=False)
                tape1.watch(conv_output)
                tape2.watch(conv_output)
                tape3.watch(conv_output)
                probability = classifier(gap_layer(conv_output))[:, 0]
                if predicted_class == 1:
                    class_score = probability
                else:
                    class_score = 1 - probability
            first_derivative = tape1.gradient(class_score, conv_output)
        second_derivative = tape2.gradient(first_derivative, conv_output)
    third_derivative = tape3.gradient(second_derivative, conv_output)
    alpha_numerator = second_derivative
    alpha_denominator = 2.0 * second_derivative + third_derivative * tf.reduce_sum(conv_output, axis=(1, 2), keepdims=True)
    alpha = alpha_numerator / (alpha_denominator + tf.keras.backend.epsilon())
    weights = tf.reduce_sum(alpha * tf.nn.relu(first_derivative), axis=(1, 2))
    heatmap = tf.reduce_sum(weights[:, None, None, :] * conv_output, axis=-1)[0]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def create_gradcam_overlay(image_path, heatmap):
    original_image = Image.open(image_path).convert('RGB')
    original_array = np.array(original_image).astype(np.float32)
    resized_heatmap = tf.image.resize(heatmap[..., np.newaxis], (original_image.height, original_image.width)).numpy().squeeze()
    jet = matplotlib.colormaps['jet']
    heatmap_rgb = jet(np.clip(resized_heatmap, 0, 1))[..., :3] * 255
    overlay = 0.6 * original_array + 0.4 * heatmap_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

def create_annotation_image(image_path, annotation_path):
    if annotation_path is None:
        return None
    original_image = Image.open(image_path).convert('RGB')
    annotation_image = original_image.copy()
    draw = ImageDraw.Draw(annotation_image)
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    xml_width = int(root.find('size/width').text)
    xml_height = int(root.find('size/height').text)
    scale_x = original_image.width / xml_width
    scale_y = original_image.height / xml_height
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        if box is None:
            continue
        xmin = int(float(box.find('xmin').text) * scale_x)
        ymin = int(float(box.find('ymin').text) * scale_y)
        xmax = int(float(box.find('xmax').text) * scale_x)
        ymax = int(float(box.find('ymax').text) * scale_y)
        line_width = max(3, int(original_image.width / 200))
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(220, 65, 65), width=line_width)
    return annotation_image

def reset_case():
    st.session_state.pop('analysis_result', None)
    st.session_state['show_visual'] = False

def toggle_visual():
    current = st.session_state.get('show_visual', False)
    st.session_state['show_visual'] = not current

def show_result_card(label, value, note):
    st.html(f'\n        <div class="result-card">\n\n            <div class="result-label">\n                {label}\n            </div>\n\n            <div class="result-value">\n                {value}\n            </div>\n\n            <div class="result-note">\n                {note}\n            </div>\n\n        </div>\n        ')
banner = base64.b64encode((IMAGES_DIR / '1.png').read_bytes()).decode()
st.html(f"""\n<div style="\n    width:100%;\n    height:260px;\n    position:relative;\n    overflow:hidden;\n    background:#021220;\n\n    background-image:\n        linear-gradient(\n            90deg,\n            rgba(2,18,32,0.94),\n            rgba(2,18,32,0.55),\n            rgba(2,18,32,0.05)\n        ),\n        url('data:image/png;base64,{banner}');\n\n    background-size:cover, contain;\n    background-position:center, right center;\n    background-repeat:no-repeat;\n">\n\n    <div style="\n        position:absolute;\n        left:65px;\n        top:50%;\n        transform:translateY(-50%);\n    ">\n\n        <div style="\n            font-family:'Segoe UI', Arial, sans-serif;\n            font-size:66px;\n            font-weight:700;\n            letter-spacing:-2px;\n            color:white;\n        ">\n            Case Explorer\n        </div>\n\n    </div>\n</div>\n""")
st.html("""
    <div class="intro-box">
        Explore how the <strong>ResNet50 model</strong> analyses chest X-ray
        cases. Select a case and run the model to view its <strong>TB
        Prediction Score</strong>, <strong>TB/Non-TB prediction</strong>,
        and known result. Grad-CAM++ can then be used to visualise the image
        regions that most influenced the model's prediction and compare them
        with the available annotation.
    </div>
    """)
st.html('<div class="soft-divider"></div>')
st.html("""
    <div class="section-title">
        Select a Case
    </div>

    <div class="section-subtitle">
        Choose a chest X-ray case to begin the analysis.
    </div>
    """)
case_names = [f'Case {index:02d}' for index in range(1, len(case_images) + 1)]
case_lookup = dict(zip(case_names, case_images))
selected_case = st.selectbox('Case', options=case_names, index=None, placeholder='Choose a case', label_visibility='collapsed', on_change=reset_case)
if selected_case is not None:
    selected_image_path = case_lookup[selected_case]
    selected_image = Image.open(selected_image_path).convert('RGB')
    left_space, image_column, right_space = st.columns([1.4, 2, 1.4])
    with image_column:
        st.html(f'\n            <div class="case-heading">\n                {selected_case}\n            </div>\n            ')
        st.image(selected_image, use_container_width=True)
        run_model = st.button('Run Model', type='primary', use_container_width=True)
    if run_model:
        with st.spinner('Analysing chest X-ray...'):
            try:
                score, prediction = predict_case(selected_image_path)
                known_label = get_known_label(selected_image_path)
                heatmap = generate_gradcam_plus_plus(selected_image_path, prediction)
                gradcam_image = create_gradcam_overlay(selected_image_path, heatmap)
                annotation_path = get_annotation_path(selected_image_path)
                if annotation_path is not None:
                    annotation_image = create_annotation_image(selected_image_path, annotation_path)
                else:
                    annotation_image = None
                st.session_state['analysis_result'] = {'case': selected_case, 'score': score, 'prediction': prediction, 'known_label': known_label, 'gradcam': gradcam_image, 'annotation': annotation_image}
                st.session_state['show_visual'] = False
            except Exception as error:
                st.error(f'The case could not be analysed: {error}')
    result = st.session_state.get('analysis_result')
    if result is not None and result['case'] == selected_case:
        score = result['score']
        prediction = result['prediction']
        known_label = result['known_label']
        if prediction == 1:
            prediction_text = 'Tuberculosis (TB)'
        else:
            prediction_text = 'Non-TB'
        if known_label == 1:
            known_text = 'Tuberculosis (TB)'
        else:
            known_text = 'Non-TB'
        if prediction == 1 and known_label == 1:
            outcome = 'True Positive'
        elif prediction == 0 and known_label == 0:
            outcome = 'True Negative'
        elif prediction == 1 and known_label == 0:
            outcome = 'False Positive'
        else:
            outcome = 'False Negative'
        correct = prediction == known_label
        st.html('<div class="soft-divider"></div>')
        st.html("""
            <div class="section-title">
                Analysis Result
            </div>

            <div class="section-subtitle">
                Review the model output and compare it with
                the known result for this case.
            </div>
            """)
        col1, col2, col3 = st.columns(3, gap='large')
        with col1:
            show_result_card('Model Prediction', prediction_text, 'ResNet50 classification')
        with col2:
            show_result_card('TB Prediction Score', f'{score * 100:.1f}%', f'Screening threshold: {THRESHOLD * 100:.2f}%')
        with col3:
            show_result_card('Known Result', known_text, 'Reference label')
        if correct:
            st.html(f'\n                <div class="correct-box">\n                    ✓ Prediction matched the known result\n                    &nbsp;&nbsp;•&nbsp;&nbsp;\n                    {outcome}\n                </div>\n                ')
        else:
            st.html(f'\n                <div class="incorrect-box">\n                    ⚠ Prediction did not match the known result\n                    &nbsp;&nbsp;•&nbsp;&nbsp;\n                    {outcome}\n                </div>\n                ')
        st.html("""
            <div style="
                margin-top:30px;
                font-family:'Segoe UI', Arial, sans-serif;
                font-size:22px;
                font-weight:650;
                color:var(--text-color);
            ">
                Visual Explanation
            </div>

            <div style="
                margin-top:5px;
                margin-bottom:14px;
                font-family:'Segoe UI', Arial, sans-serif;
                font-size:14.5px;
                line-height:1.65;
                color:var(--text-color);
                opacity:0.70;
            ">
                View the original chest X-ray alongside the
                Grad-CAM++ visualisation and available annotation.
            </div>
            """)
        visual_open = st.session_state.get('show_visual', False)
        if visual_open:
            button_text = 'Hide Visual Explanation'
        else:
            button_text = 'View Visual Explanation'
        st.button(button_text, key='visual_button', use_container_width=True, on_click=toggle_visual)
        if st.session_state.get('show_visual', False):
            st.html('<div class="soft-divider"></div>')
            st.html("""
                <div class="section-title">
                    Visual Explanation
                </div>

                <div class="section-subtitle">
                    Compare the original image, model attention
                    and the available labelled region.
                </div>
                """)
            original_col, gradcam_col, annotation_col = st.columns(3, gap='large')
            with original_col:
                with st.container(border=True, height=440):
                    st.markdown('#### Original X-ray')
                    st.image(selected_image, use_container_width=True)
                    st.caption('Original chest X-ray used as model input.')
            with gradcam_col:
                with st.container(border=True, height=440):
                    st.markdown('#### Grad-CAM++')
                    st.image(result['gradcam'], use_container_width=True)
                    st.caption('Visualisation of regions influencing the model prediction.')
            with annotation_col:
                with st.container(border=True, height=440):
                    st.markdown('#### Annotation')
                    if result['annotation'] is not None:
                        st.image(result['annotation'], use_container_width=True)
                        st.caption('Dataset-labelled region provided for visual comparison.')
                    else:
                        st.markdown("""
                            <div style="
                                height:280px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                text-align:center;
                                padding:25px;
                                font-family:'Segoe UI', Arial, sans-serif;
                                font-size:14px;
                                line-height:1.6;
                                color:var(--text-color);
                                opacity:0.58;
                            ">
                                No annotation is available for this case.
                            </div>
                            """, unsafe_allow_html=True)
            st.html("""
                <div class="visual-info">

                    <div class="visual-info-title">
                        Understanding the Visualisation
                    </div>

                    <div class="visual-info-text">
                        The Grad-CAM++ heatmap indicates the image regions
                        that most influenced the model's prediction. Warmer
                        colours, such as red and yellow, represent areas of
                        stronger influence, while cooler colours, such as
                        blue, represent areas of weaker influence. The
                        annotation image displays the dataset's original
                        labelled region and is provided for visual comparison.
                        Together, these visualisations offer insight into the
                        model's decision-making process. However, the heatmap
                        indicates influential regions only and does not confirm
                        the presence of a tuberculosis lesion.
                    </div>

                </div>
                """)
st.html('<div class="soft-divider"></div>')
st.html("""
    <div class="disclaimer-box">

        <div class="disclaimer-title">
            Disclaimer
        </div>

        <div class="disclaimer-text">
            CXR Insight is a research and educational demonstration only,
            not a clinical diagnostic tool, and its predictions should not
            be used to inform real medical decisions.
        </div>

    </div>
    """)
