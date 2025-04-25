
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration
from keybert import KeyBERT
import spacy
import io
import base64
import datetime

st.set_page_config(page_title="AI Image Captioner", layout="centered")
st.title("📸 Image Captioning + Keywords Generator (Streamlit Version)")

@st.cache_resource
def load_models():
    import spacy.cli
    spacy.cli.download("en_core_web_sm")  # ✅ เพิ่มบรรทัดนี้
    nlp = spacy.load("en_core_web_sm")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    kw_model = KeyBERT()
    return nlp, processor, model, kw_model

nlp, processor, model, kw_model = load_models()

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def extract_noun_keywords(text, top_n=50):
    raw_keywords = kw_model.extract_keywords(text, top_n=top_n*2)
    keyword_list = list(set([kw[0] for kw in raw_keywords]))
    noun_keywords = []
    for kw in keyword_list:
        doc = nlp(kw)
        if any(token.pos_ == "NOUN" for token in doc):
            noun_keywords.append(kw)
    return noun_keywords[:top_n]

def embed_text(image, title, keywords):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), "Title: " + title, fill="black", font=font)
    draw.text((10, 40), "Keywords: " + ", ".join(keywords[:6]), fill="gray", font=font)
    return image

def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write("### Results")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        caption = generate_caption(image)
        keywords = extract_noun_keywords(caption)
        result_img = embed_text(image.copy(), caption, keywords)

        st.image(result_img, caption=f"📝 {caption}", use_column_width=True)
        img_bytes = image_to_bytes(result_img)
        filename = f"{caption.replace(' ', '_')[:50]}.jpg"
        st.download_button(label="⬇️ Download Image", data=img_bytes, file_name=filename, mime="image/jpeg")
