import streamlit as st
from pathlib import Path
from PIL import Image
import base64
import numpy as np
import tensorflow as tf
import matplotlib
st.set_page_config(page_title='Upload & Analyse | CXR Insight', page_icon='🫁', layout='wide')
BASE_DIR = Path(__file__).resolve().parents[1]
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

.page-title {
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


/* Sections */

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


/* Prediction */

.prediction-card {
    max-width: 620px;
    margin: 0 auto;
    padding: 26px 30px;

    border: 1px solid rgba(120, 150, 170, 0.25);
    border-top: 3px solid #2FA7E0;
    border-radius: 13px;

    background: var(--secondary-background-color);
    text-align: center;

    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.025);
}

.prediction-label {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    opacity: 0.58;
    margin-bottom: 10px;
}

.prediction-value {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 30px;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 16px;
}

.score-label {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14px;
    opacity: 0.68;
    margin-bottom: 4px;
}

.score-value {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 25px;
    font-weight: 700;
    color: #2FA7E0;
}


/* Visualisation */

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


/* Upload box */

[data-testid="stFileUploader"] {
    margin-bottom: 8px;
}

/* Hide the + Add files button */

[data-testid="stFileUploader"] button[aria-label="Add files"],
[data-testid="stFileUploader"] button[aria-label="Add file"],
[data-testid="stFileUploader"] button[title="Add files"],
[data-testid="stFileUploader"] button[title="Add file"] {
    display: none !important;
}

/* Upload box */

[data-testid="stFileUploader"] {
    margin-bottom: 8px;
}


/* Hide 200MB / PNG / JPG text */

[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
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
banner = base64.b64encode((IMAGES_DIR / '1.png').read_bytes()).decode()
st.html(f"""\n<div style="\n    width:100%;\n    height:260px;\n    position:relative;\n    overflow:hidden;\n    background:#021220;\n\n    background-image:\n        linear-gradient(\n            90deg,\n            rgba(2,18,32,0.94),\n            rgba(2,18,32,0.55),\n            rgba(2,18,32,0.05)\n        ),\n        url('data:image/png;base64,{banner}');\n\n    background-size:cover, contain;\n    background-position:center, right center;\n    background-repeat:no-repeat;\n">\n\n    <div style="\n        position:absolute;\n        left:65px;\n        top:50%;\n        transform:translateY(-50%);\n    ">\n\n        <div style="\n            font-family:'Segoe UI', Arial, sans-serif;\n            font-size:66px;\n            font-weight:700;\n            letter-spacing:-2px;\n            color:white;\n        ">\n            Upload &amp; Analyse\n        </div>\n\n    </div>\n</div>\n""")

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

def prepare_image(image):
    image = image.convert('RGB')
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, IMAGE_SIZE)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor

def predict_image(image):
    image_tensor = prepare_image(image)
    output = model(image_tensor, training=False)
    score = float(tf.reshape(output, [-1])[0].numpy())
    prediction = int(score >= THRESHOLD)
    return (score, prediction)

def generate_gradcam_plus_plus(image, predicted_class):
    image = image.convert('RGB')
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, IMAGE_SIZE)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensor = tf.keras.applications.resnet50.preprocess_input(image_tensor)
    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_output = feature_model(image_tensor, training=False)
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

def create_gradcam_overlay(image, heatmap):
    original_image = image.convert('RGB')
    original_array = np.array(original_image).astype(np.float32)
    resized_heatmap = tf.image.resize(heatmap[..., np.newaxis], (original_image.height, original_image.width)).numpy().squeeze()
    jet = matplotlib.colormaps['jet']
    heatmap_rgb = jet(np.clip(resized_heatmap, 0, 1))[..., :3] * 255
    overlay = 0.6 * original_array + 0.4 * heatmap_rgb
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

def check_image(image):
    image_rgb = image.convert('RGB')
    array = np.array(image_rgb).astype(np.float32)
    width, height = image_rgb.size
    dimensions_ok = width >= 224 and height >= 224
    brightness = float(np.mean(array))
    contrast = float(np.std(array))
    brightness_ok = 10 <= brightness <= 245
    contrast_ok = contrast >= 10
    not_blank = contrast >= 5
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    channel_difference = float(np.mean(np.abs(red - green) + np.abs(red - blue) + np.abs(green - blue)) / 3)
    grayscale_like = channel_difference < 12
    suitable = dimensions_ok and brightness_ok and contrast_ok and not_blank and grayscale_like
    return {'dimensions_ok': dimensions_ok, 'brightness_ok': brightness_ok, 'contrast_ok': contrast_ok, 'not_blank': not_blank, 'grayscale_like': grayscale_like, 'suitable': suitable}

def reset_analysis():
    st.session_state.pop('upload_result', None)
    st.session_state['show_upload_visual'] = False

def toggle_visual():
    current = st.session_state.get('show_upload_visual', False)
    st.session_state['show_upload_visual'] = not current
st.html("""
<div class="intro-box">
    Upload a frontal chest X-ray to analyse it using the
    <strong>ResNet50 model</strong>. The model will provide a
    <strong>TB Prediction Score</strong> and
    <strong>TB/Non-TB prediction</strong>. Grad-CAM++ can then
    be used to visualise the image regions that most influenced
    the model's output.
</div>
""")
st.html('<div class="soft-divider"></div>')
st.html("""
<div class="section-title">
    Upload Chest X-ray
</div>

<div class="section-subtitle">
    Upload a frontal chest X-ray in PNG, JPG or JPEG format.
</div>
""")
uploaded_file = st.file_uploader('Upload Chest X-ray', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False, label_visibility='collapsed', on_change=reset_analysis, key='uploaded_xray')
if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file).convert('RGB')
    except Exception:
        st.error('The uploaded file could not be opened as an image.')
        st.stop()
    checks = check_image(uploaded_image)
    left, image_col, right = st.columns([1.4, 2, 1.4])
    with image_col:
        st.image(uploaded_image, use_container_width=True)
    with st.expander('Image checks'):
        if checks['dimensions_ok']:
            st.success('Image dimensions are suitable.')
        else:
            st.error('Image dimensions are too small.')
        if checks['not_blank']:
            st.success('The image is not blank.')
        else:
            st.error('The image appears blank.')
        if checks['brightness_ok'] and checks['contrast_ok']:
            st.success('Brightness and contrast are suitable.')
        else:
            st.error('Brightness or contrast is unsuitable.')
        if checks['grayscale_like']:
            st.success('The image has predominantly grayscale characteristics.')
        else:
            st.error('The image appears to be coloured and is not suitable for analysis.')
        st.caption('These checks assess basic image properties only. They do not verify that the uploaded image is a chest X-ray.')
    confirmed = st.checkbox('I confirm that this is a frontal chest X-ray.')
    run_model = st.button('Run Model', type='primary', use_container_width=True, disabled=not confirmed or not checks['suitable'])
    if not checks['suitable']:
        st.warning('The image does not meet the basic technical requirements for analysis.')
    if run_model:
        with st.spinner('Analysing chest X-ray...'):
            try:
                score, prediction = predict_image(uploaded_image)
                heatmap = generate_gradcam_plus_plus(uploaded_image, prediction)
                gradcam_image = create_gradcam_overlay(uploaded_image, heatmap)
                st.session_state['upload_result'] = {'score': score, 'prediction': prediction, 'gradcam': gradcam_image}
                st.session_state['show_upload_visual'] = False
            except Exception as error:
                st.error(f'The image could not be analysed: {error}')
    result = st.session_state.get('upload_result')
    if result is not None:
        score = result['score']
        prediction = result['prediction']
        prediction_text = 'Tuberculosis (TB)' if prediction == 1 else 'Non-TB'
        st.html('<div class="soft-divider"></div>')
        st.html("""
        <div class="section-title">
            Analysis Result
        </div>

        <div class="section-subtitle">
            Model output for the uploaded chest X-ray.
        </div>
        """)
        st.html(f'\n            <div class="prediction-card">\n\n                <div class="prediction-label">\n                    Model Prediction\n                </div>\n\n                <div class="prediction-value">\n                    {prediction_text}\n                </div>\n\n                <div class="score-label">\n                    TB Prediction Score\n                </div>\n\n                <div class="score-value">\n                    {score * 100:.1f}%\n                </div>\n\n            </div>\n            ')
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
            View the uploaded chest X-ray alongside the
            Grad-CAM++ visualisation.
        </div>
        """)
        visual_open = st.session_state.get('show_upload_visual', False)
        button_text = 'Hide Visual Explanation' if visual_open else 'View Visual Explanation'
        st.button(button_text, key='upload_visual_button', use_container_width=True, on_click=toggle_visual)
        if st.session_state.get('show_upload_visual', False):
            st.html('<div class="soft-divider"></div>')
            original_col, gradcam_col = st.columns(2, gap='large')
            with original_col:
                with st.container(border=True, height=450):
                    st.markdown('#### Original X-ray')
                    st.image(uploaded_image, use_container_width=True)
                    st.caption('Original chest X-ray submitted for analysis.')
            with gradcam_col:
                with st.container(border=True, height=450):
                    st.markdown('#### Grad-CAM++')
                    st.image(result['gradcam'], use_container_width=True)
                    st.caption('Visualisation of regions influencing the model prediction.')
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
                    visualisation offers insight into the model's
                    decision-making process. However, the heatmap
                    indicates influential regions only and does not
                    confirm the presence or location of a tuberculosis
                    lesion.
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
