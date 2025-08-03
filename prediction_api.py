import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import os
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import CORS Middleware
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Define paths to saved models and encoders
MODELS_DIR = 'saved_models'
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_svm_model.joblib')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_xgboost_model.joblib')
LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_logreg_model.joblib')
LE_PATH = os.path.join(MODELS_DIR, 'label_encoder.joblib')

# --- Pydantic Model for Request Body ---
class IntentRequest(BaseModel):
    text: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Real Estate Intent Recognition API",
    description="An API to predict the intent of a sentence using three different machine learning models.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
# This is the new section that fixes the error.
# It allows web pages from any origin to connect to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- Global variables to hold loaded models ---
# These will be loaded once at startup
models = {}

# --- Helper Function for BERT Embedding ---
def get_bert_embedding(text: str, model, tokenizer):
    """Generates a BERT embedding for a single piece of text."""
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1)

# --- API Events ---
@app.on_event("startup")
def load_models():
    """
    Load all machine learning models and encoders into memory at startup.
    This is efficient as it's done only once.
    """
    print("Loading models and encoders at startup...")
    try:
        models['svm'] = joblib.load(SVM_MODEL_PATH)
        models['xgb'] = joblib.load(XGB_MODEL_PATH)
        models['logreg'] = joblib.load(LOGREG_MODEL_PATH)
        models['label_encoder'] = joblib.load(LE_PATH)
        models['bert_tokenizer'] = BertTokenizer.from_pretrained('bert-base-uncased')
        models['bert_model'] = BertModel.from_pretrained('bert-base-uncased')
        print("All models loaded successfully! 🚀")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: A required model file was not found: {e}")
        # In a real application, you might want to handle this more gracefully
        # For now, we'll exit if a model is missing.
        exit()

# --- API Endpoints ---
@app.get("/", summary="Root endpoint", description="A simple hello world endpoint to confirm the API is running.")
def read_root():
    return {"message": "Welcome to the Intent Recognition API. Please use the /predict endpoint to get started."}

@app.post("/predict", summary="Predict Intent", description="Takes a sentence and returns the predicted intent from three different models.")
def predict_intent(request: IntentRequest):
    """
    Predicts the intent of the input text using the pre-loaded models.
    """
    # Generate embedding for the input sentence
    embedding = get_bert_embedding(request.text, models['bert_model'], models['bert_tokenizer'])

    # --- Get predictions from each model ---
    pred_svm_idx = models['svm'].predict(embedding)[0]
    pred_xgb_idx = models['xgb'].predict(embedding)[0]
    pred_logreg_idx = models['logreg'].predict(embedding)[0]

    # --- Inverse transform the predictions to get the original labels ---
    label_encoder = models['label_encoder']
    pred_svm_label = label_encoder.inverse_transform([pred_svm_idx])[0]
    pred_xgb_label = label_encoder.inverse_transform([pred_xgb_idx])[0]
    pred_logreg_label = label_encoder.inverse_transform([pred_logreg_idx])[0]

    # --- Return the results in a JSON response ---
    return {
        "input_text": request.text,
        "predictions": {
            "svm_prediction": pred_svm_label,
            "xgboost_prediction": pred_xgb_label,
            "logistic_regression_prediction": pred_logreg_label
        }
    }

# --- Run the API ---
if __name__ == '__main__':
    # To run this API, save the file as `predict_intent.py` and run the following command in your terminal:
    # uvicorn predict_intent:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
