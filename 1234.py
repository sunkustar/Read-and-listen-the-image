import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from gtts import gTTS
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Load image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Translator instance
translator = Translator()

# Languages for translation (Indian languages)
languages = ["hi", "ta", "bn", "te", "kn"]  # Hindi, Tamil, Bengali, Telugu, Kannada

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def translate_caption(caption, lang):
    translated = translator.translate(caption, dest=lang)
    return translated.text

def text_to_speech(caption, lang):
    tts = gTTS(text=caption, lang=lang)
    audio_path = "caption.mp3"
    tts.save(audio_path)
    return audio_path

# Streamlit UI
st.title("Image Captioning, Translation, and Audio")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate caption
    caption = generate_caption(image)
    st.subheader("Generated Caption")
    st.write(caption)

    # Select language for translation
    language = st.selectbox("Select language for translation", languages)

    # Translate caption
    translated_caption = translate_caption(caption, language)
    st.subheader(f"Translated Caption ({language})")
    st.write(translated_caption)

    # Generate audio
    audio_path = text_to_speech(translated_caption, language)
    st.subheader("Audio of the caption")
    st.audio(audio_path, format="audio/mp3")

    # Option to download the audio file
    st.download_button("Download Audio", audio_path, file_name="caption_audio.mp3")
