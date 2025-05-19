import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.nn.functional import softmax
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_path = "/Users/vijay/ner-nlp/ner-course-model-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
label_map = model.config.id2label

# Function to get predictions
def predict(text):
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
    offset_mapping = encoding.pop("offset_mapping")  # ✅ Remove before passing to model

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    probs = softmax(logits, dim=-1)
    predictions = torch.argmax(probs, dim=-1).squeeze().tolist()
    offset_mapping = offset_mapping.squeeze().tolist()

    entities = []
    entity = None
    for idx, label_id in enumerate(predictions):
        label = label_map[label_id]
        start, end = offset_mapping[idx]
        if start == end:
            continue

        if label.startswith("B-"):
            if entity:
                entities.append(entity)
            entity = {
                "label": label[2:],
                "start": start,
                "end": end,
                "text": text[start:end]
            }
        elif label.startswith("I-") and entity and entity["label"] == label[2:]:
            entity["end"] = end
            entity["text"] = text[entity["start"]:entity["end"]]
        else:
            if entity:
                entities.append(entity)
                entity = None
    if entity:
        entities.append(entity)

    return entities


# Streamlit UI
st.title("📘 Course Catalog NER Annotator")
st.markdown("Enter a course description to extract entities like CourseCode, Title, Credit, Instructor, etc.")

user_input = st.text_area("✍️ Enter course text here:", height=200)

if st.button("Run Prediction") and user_input:
    with st.spinner("Predicting..."):
        predictions = predict(user_input)
    st.subheader("📌 Extracted Entities")
    if predictions:
        for ent in predictions:
            st.markdown(f"**{ent['label']}**: `{ent['text']}`")
    else:
        st.info("No entities found.")
