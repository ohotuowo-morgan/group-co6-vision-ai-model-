# STEP 1: Import Required Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO

# STEP 2: Page config
st.set_page_config(
    page_title="Dog vs Wolf Classifier",
    page_icon="https://cdn-icons-png.flaticon.com/128/17839/17839240.png",
    layout="centered",
)

# STEP 3: Techy custom CSS (structure/layout only)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

.hero {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}
.hero h1 {
    font-size: 2.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.hero p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    opacity: 0.7;
}

.card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px;
}

/* FIXED BUTTON CSS */
div[data-testid="stButton"] {
    display: flex;
    justify-content: center;
    width: 100%;
}
div[data-testid="stButton"] button {
    width: 100% !important;
    max-width: 400px !important;
    min-height: 52px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    padding: 0.85rem 2rem !important; /* Fixed padding: Top/Bottom 0.85rem, Left/Right 2rem */
    transition: transform 0.15s ease !important;
}
div[data-testid="stButton"] button:hover {
    transform: scale(1.02) !important;
}
div[data-testid="stButton"] button:active {
    transform: scale(0.98) !important;
}

.result-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    text-align: center;
    padding: 0.6rem;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.3);
    margin-bottom: 1rem;
}

.centered-img {
    display: flex;
    justify-content: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# STEP 4: Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "mobilenetv3_transfer.keras"
    )
    return model

# STEP 5: Prediction function
def predict(model, pil_image):
    img = pil_image.convert("RGB").resize((128, 128))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    preds = model.predict(arr, verbose=0)[0]
    prob_dog = float(preds[0])
    prob_wolf = float(preds[1])
    label = "Dog" if prob_dog >= prob_wolf else "Wolf"
    return label, prob_wolf * 100, prob_dog * 100

# STEP 6: Hero / welcome section
st.markdown("""
<div class="hero">
    <h1>Dog vs Wolf Classifier</h1>
    <p>GET-324 · TRANSFER LEARNING · Predictive AI Model</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
This app uses a fine-tuned <b>MobileNetV3</b> convolutional neural network
to tell the difference between dogs and wolves from a single image.
Drop a photo below, hit <b>Predict</b>, and the model will return its verdict
with confidence scores for each class.
</div>
""", unsafe_allow_html=True)

model = load_model()

# STEP 7: Session state so predictions don't fire on upload
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None
if "result" not in st.session_state:
    st.session_state.result = None

uploaded_file = st.file_uploader("Drop a dog or wolf image", type=["jpg", "jpeg", "png"])

# Reset stored result whenever a new/different file is uploaded
if uploaded_file is not None and uploaded_file.file_id != st.session_state.last_file_id:
    st.session_state.last_file_id = uploaded_file.file_id
    st.session_state.result = None

if uploaded_file:
    img = Image.open(uploaded_file)

    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<div class="centered-img"><img src="data:image/png;base64,{b64_img}" width="320" style="border-radius:12px;"></div>',
        unsafe_allow_html=True,
    )

    # FIXED: Added native Streamlit full-width and primary style flags
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            label, wolf_pct, dog_pct = predict(model, img)
            st.session_state.result = (label, wolf_pct, dog_pct)

    # STEP 8: Show results only after Predict has been clicked
    if st.session_state.result:
        label, wolf_pct, dog_pct = st.session_state.result
        badge_class = "badge-wolf" if label == "Wolf" else "badge-dog"
        st.markdown(
            f'<div class="result-badge {badge_class}">Prediction: {label}</div>',
            unsafe_allow_html=True,
        )
        st.progress(int(dog_pct), text=f"Dog: {dog_pct:.1f}%")
        st.progress(int(wolf_pct), text=f"Wolf: {wolf_pct:.1f}%")
else:
    st.session_state.result = None
