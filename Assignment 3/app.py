import os
import streamlit as st
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

# Suppress TensorFlow oneDNN warnings:

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Page config:

st.set_page_config(
    page_title="IMDB SENTIMENT ANALYSIS",
    layout="centered"
)

st.title("IMDB SENTIMENT ANALYSIS")
st.write(
    "This demo web app uses a **fine-tuned RoBERTa model** trained on the IMDB movie reviews dataset "
    "to predict whether a review expresses **positive** or **negative** sentiment."
)

# Model definition:

class RobertaSentimentClassifier(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # CLS pooling
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits


# Load tokenizer & model:

@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Resolve project root and model path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODEL_PATH = PROJECT_ROOT / "Assignment 2" / "best_model.pt"

    model = RobertaSentimentClassifier(dropout=0.1)
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# Take in User input:

st.subheader("Please Enter a Movie Review here:")

user_text = st.text_area(
    "Type or paste a movie review below:",
    height=180,
    placeholder="This movie was absolutely fantastic. The acting was brilliant..."
)

# Prediction:

if st.button("Analyze Sentiment of this Review"):
    if user_text.strip() == "":
        st.warning("Please enter a review before running the analysis.")
    else:
        with st.spinner("Running analysis"):
            encoded = tokenizer(
                user_text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            with torch.no_grad():
                logits = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )

                # Post-hoc temperature scaling (no retraining)
                TEMPERATURE = 2.5
                probs = F.softmax(logits / TEMPERATURE, dim=1)

            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()

            # Logit margin = decision strength
            logit_margin = torch.abs(logits[0][0] - logits[0][1]).item()

            sentiment = "POSITIVE" if pred_label == 1 else "NEGATIVE"

        st.subheader("Prediction Result:")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Calibrated Confidence:** {confidence:.2%}")
        st.write(f"**Decision Strength:** {logit_margin:.2f}")


        # Neutral / Borderline Warning:
        
        if logit_margin < 1.5:
            st.warning(
                "**Neutral / Borderline Review Detected**\n\n"
                "This review expresses mixed or weak sentiment. "
                "As the IMDB dataset contains only binary labels which are only positive or negative,"
                "the model is forced to choose between positive and negative."
            )
        elif logit_margin < 3.0:
            st.info(
                "**Weak Sentiment**\n\n"
                "The sentiment signal is present but not strong."
            )

# Footer:

st.markdown("---------------------")
st.caption(
    "Fine-tuned RoBERTa Model for Movie Analysis | IMDB Dataset | Bring Your Own Method (NLP - Sentiment Analysis) | Applied Deep Learning (WS 2025)"
)
