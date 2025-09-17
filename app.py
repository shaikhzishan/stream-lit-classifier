import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

# --- Page config ---
st.set_page_config(page_title="Universal Image Classifier", layout="wide")

# --- Helper functions ---

def load_css(file_name: str):
    """Load CSS file into Streamlit if exists (silent fallback)."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # No crash if CSS missing — optional console warning
        print(f"Warning: CSS file '{file_name}' not found. Using default styles.")

@st.cache_resource
def load_model():
    """Load MobileNetV2 model (pretrained on ImageNet)."""
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

def pil_to_model_array(image: Image.Image, target_size=(224, 224)):
    """Convert PIL image to a float32 numpy array suitable for the model."""
    img = image.convert("RGB").resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)  # float32
    return arr  # shape (224,224,3)

def build_augmentation_pipeline():
    """Create a stateless augmentation pipeline using Keras preprocessing layers."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.12),   # ~±12% of a full circle (about ±43 degrees)
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ])

def predict_with_augmentations(model, pil_image: Image.Image, num_augmentations: int = 3):
    """
    Apply `num_augmentations` random transforms to the input image, run model predictions
    on all augmented copies, and return:
      - averaged_probs: numpy array shape (1000,) averaged over augmentations
      - augmented_images: numpy array shape (num_augmentations, 224, 224, 3) in uint8 for display
    """
    # Convert PIL to base array and create a batch of repeated copies
    base_arr = pil_to_model_array(pil_image)  # (224,224,3)
    base_batch = np.expand_dims(base_arr, axis=0).astype(np.float32)  # (1,224,224,3)
    base_batch_tf = tf.convert_to_tensor(base_batch)

    # Repeat to create a batch of size `num_augmentations`
    batch = tf.repeat(base_batch_tf, repeats=num_augmentations, axis=0)  # (N,224,224,3)

    # Build augmentation pipeline and apply it
    aug_pipeline = build_augmentation_pipeline()
    augmented_batch = aug_pipeline(batch, training=True)  # random transforms per example

    # Keep a displayable copy (uint8 0-255) for showing augmented images
    # augmented_batch is float32 in [0,255) or possibly outside due to contrast; clip & convert
    augmented_display = tf.clip_by_value(augmented_batch, 0.0, 255.0)
    augmented_display_uint8 = tf.cast(augmented_display, tf.uint8).numpy()  # (N,224,224,3)

    # Preprocess for MobileNetV2 (expects float in -1..1)
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_batch.numpy())  # (N,224,224,3)

    # Run predictions on the whole batch
    preds = model.predict(processed, verbose=0)  # (N,1000)

    # Average probabilities across augmentations
    averaged_probs = np.mean(preds, axis=0)  # (1000,)

    return averaged_probs, augmented_display_uint8

def decode_top_k_from_probs(probs: np.ndarray, top_k: int = 3):
    """
    Given a 1D probs array (length 1000 from ImageNet), return top_k decoded predictions
    in the format [(imagenet_id, label, score), ...]
    """
    # decode_predictions expects a batch shape (1,1000)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(np.expand_dims(probs, axis=0), top=top_k)[0]
    return decoded

# --- Main App UI & Logic ---

# Load CSS (optional)
load_css("style.css")

# Load model (cached)
model = load_model()

# Title and description (brief)
st.title("🖼️ Universal Image Classifier with Augmentation")
st.write(
    "Upload an image to get top-3 ImageNet predictions. "
    "The app applies random augmentations at prediction time and averages results for robustness."
)

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    num_augmentations = st.slider("Number of augmentations to use", 1, 8, value=3, help="How many randomly-augmented copies to generate and predict on (predictions averaged).")
    show_augmented = st.checkbox("Show augmented images", value=True)
    show_original_prediction = st.checkbox("Also show prediction on original image (no augmentation)", value=True)
    st.markdown("---")
    st.markdown("Model: **MobileNetV2 (ImageNet)**")
    st.markdown("Input images are resized to 224×224 and preprocessed for MobileNetV2.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and display original
    image = Image.open(uploaded_file)
    st.subheader("Original image")
    st.image(image, use_column_width=True, caption="Uploaded original image")

    # Prediction on original (no augmentation) if requested
    if show_original_prediction:
        with st.spinner("Classifying original image..."):
            arr = pil_to_model_array(image)
            proc = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(arr, axis=0))
            preds_orig = model.predict(proc, verbose=0)  # (1,1000)
            decoded_orig = tf.keras.applications.mobilenet_v2.decode_predictions(preds_orig, top=3)[0]

        st.subheader("Top-3 (original image)")
        for i, (imagenet_id, label, score) in enumerate(decoded_orig):
            label_readable = label.replace("_", " ").title()
            st.write(f"{i+1}. **{label_readable}** — {score:.2%}")

    # Prediction with augmentations
    with st.spinner(f"Generating {num_augmentations} augmentations and classifying..."):
        averaged_probs, augmented_images = predict_with_augmentations(model, image, num_augmentations=num_augmentations)
        decoded_avg = decode_top_k_from_probs(averaged_probs, top_k=3)

    st.subheader(f"Top-3 (averaged over {num_augmentations} augmentation{'s' if num_augmentations>1 else ''})")
    for i, (imagenet_id, label, score) in enumerate(decoded_avg):
        label_readable = label.replace("_", " ").title()
        st.write(f"{i+1}. **{label_readable}** — {score:.2%}")

    # Optionally show augmented images in a grid
    if show_augmented:
        st.subheader("Augmented images used for prediction")
        cols = st.columns(min(4, num_augmentations))
        for idx in range(num_augmentations):
            col = cols[idx % len(cols)]
            # augmented_images is numpy uint8 (N,224,224,3)
            aug_img_arr = augmented_images[idx]
            # Convert numpy to PIL for caption-friendly display
            aug_pil = Image.fromarray(aug_img_arr)
            col.image(aug_pil, use_column_width=True, caption=f"Augmented #{idx+1}")

# Footer / usage tip
st.markdown("---")
st.caption("Tip: Increasing augmentation count can stabilize predictions but increases inference time. This demo applies augmentation at prediction time for demonstration and educational purposes.")
