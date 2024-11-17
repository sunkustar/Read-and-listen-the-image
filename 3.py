import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from mosestokenizer import MosesSentenceSplitter
from nltk import sent_tokenize
from indicnlp.tokenize.sentence_tokenize import sentence_split, DELIM_PAT_NO_DANDA
import streamlit as st

# Initialize model and tokenizer directories
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"
indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-1B"
indic_indic_ckpt_dir = "ai4bharat/indictrans2-indic-indic-dist-320M"
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Language codes for translation
flores_codes = {
    "hin_Deva": "hi", "tel_Telu": "te", "tam_Taml": "ta", "ben_Beng": "bn",
}

# Sentence splitting function
def split_sentences(input_text, lang):
    if lang == "eng_Latn":
        input_sentences = sent_tokenize(input_text)
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        if len(sents_nltk) < len(sents_moses):
            input_sentences = sents_nltk
        else:
            input_sentences = sents_moses
        input_sentences = [sent.replace("\xad", "") for sent in input_sentences]
    else:
        input_sentences = sentence_split(input_text, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA)
    return input_sentences

# Model and tokenizer initialization
def initialize_model_and_tokenizer(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, trust_remote_code=True, low_cpu_mem_usage=True)
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model

# Batch translation function
def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i: i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, max_length=256, num_beams=5, num_return_sequences=1)

        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
        del inputs
        torch.cuda.empty_cache()

    return translations

# Translate the full paragraph
def translate_paragraph(input_text, src_lang, tgt_lang, model, tokenizer, ip):
    input_sentences = split_sentences(input_text, src_lang)
    translated_text = batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip)
    return " ".join(translated_text)

# Initialize the IndicProcessor and models
ip = IndicProcessor(inference=True)
en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir)
indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(indic_en_ckpt_dir)
indic_indic_tokenizer, indic_indic_model = initialize_model_and_tokenizer(indic_indic_ckpt_dir)

# Streamlit UI
st.title("IndicTrans Translation App")

# Language options for translation
tgt_lang = st.selectbox("Select Target Language", ["Hindi", "Telugu", "Tamil", "Bengali"])

# Input text area
input_text = st.text_area("Enter text to translate", "")

# Translate button
if st.button("Translate"):
    if tgt_lang == "Hindi":
        translated_text = translate_paragraph(input_text, "eng_Latn", "hin_Deva", en_indic_model, en_indic_tokenizer, ip)
    elif tgt_lang == "Telugu":
        translated_text = translate_paragraph(input_text, "eng_Latn", "tel_Telu", en_indic_model, en_indic_tokenizer, ip)
    elif tgt_lang == "Tamil":
        translated_text = translate_paragraph(input_text, "eng_Latn", "tam_Taml", en_indic_model, en_indic_tokenizer, ip)
    elif tgt_lang == "Bengali":
        translated_text = translate_paragraph(input_text, "eng_Latn", "ben_Beng", en_indic_model, en_indic_tokenizer, ip)

    # Display translated text
    st.subheader("Translated Text:")
    st.write(translated_text)
