import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Corn Leaf Disease Detection",
    page_icon="🌽",
    layout="centered"
)

# =========================================
# CUSTOM DESIGN
# =========================================
st.markdown("""
<style>

.main {
    background-color: #f5fff5;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2e7d32;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    margin-top: 20px;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================
st.markdown(
    '<div class="title">🌽 Corn Leaf Disease Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Enabled Agricultural Farmer Decision Support System</div>',
    unsafe_allow_html=True
)

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "MobileNetV2.h5",
        compile=False
    )

    return model

model = load_model()

# =========================================
# FILE UPLOADER
# =========================================
uploaded_file = st.file_uploader(
    "📤 Upload Corn Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================
# PREDICTION FUNCTION
# =========================================
def predict_image(image):

    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = float(prediction[0][0])

    # =========================================
    # CLASSIFICATION
    # =========================================
    if confidence > 0.5:

        predicted_class = "Healthy"

        healthy_score = confidence * 100

        diseased_score = (1 - confidence) * 100

        confidence_score = healthy_score

    else:

        predicted_class = "Diseased"

        diseased_score = (1 - confidence) * 100

        healthy_score = confidence * 100

        confidence_score = diseased_score

    return (
        predicted_class,
        confidence_score,
        healthy_score,
        diseased_score
    )

# =========================================
# DISPLAY RESULT
# =========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Corn Leaf Image",
        use_column_width=True
    )

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing Image..."):

            (
                predicted_class,
                confidence_score,
                healthy_score,
                diseased_score
            ) = predict_image(image)

        # =========================================
        # RESULT DISPLAY
        # =========================================
        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.success(f"Prediction: {predicted_class}")

        st.info(f"Confidence Score: {confidence_score:.2f}%")

        st.write("🤖 Model Used: MobileNetV2")

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # =========================================
        # GRAPH
        # =========================================
        st.subheader("📊 Prediction Confidence")

        labels = [
            "Healthy",
            "Diseased"
        ]

        values = [
            healthy_score,
            diseased_score
        ]

        fig, ax = plt.subplots()

        ax.bar(labels, values)

        ax.set_ylabel("Confidence (%)")

        ax.set_title("Prediction Result")

        st.pyplot(fig)

# =========================================
# FOOTER
# =========================================
st.markdown("---")

st.caption(
    "AI-Enabled Agricultural Farmer Record and Predictive "
    "Decision Support System for Lemery, Iloilo"
)
