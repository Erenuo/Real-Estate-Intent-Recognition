import joblib
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import os
import warnings

# Suppress warnings from XGBoost and other libraries for a cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
MODELS_DIR = 'saved_models'
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_svm_model.joblib')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'bert_xgboost_model.joblib')
LE_PATH = os.path.join(MODELS_DIR, 'label_encoder.joblib')

# --- Helper Function for BERT Embedding ---
def get_bert_embedding(text, model, tokenizer):
    """Generates a BERT embedding for a single piece of text."""
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Tokenize the text and create tensors
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get the model output without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Use the mean of the last hidden state as the sentence embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # Reshape for a single prediction (required by sklearn models)
    return embedding.reshape(1, -1)

# --- Main Application ---
def run_intent_recognizer():
    """
    Loads all necessary components and runs the interactive prediction loop.
    """
    # 1. Load all required files
    print("Loading models and encoders... This may take a moment.")
    try:
        model_svm = joblib.load(SVM_MODEL_PATH)
        model_xgb = joblib.load(XGB_MODEL_PATH)
        label_encoder = joblib.load(LE_PATH)
        
        # Load BERT components
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        print("All models loaded successfully! 🚀")
    except FileNotFoundError as e:
        print(f"\nError: A required file was not found: {e}")
        print("Please make sure you have run all the necessary training scripts and that the following files exist:")
        print(f"- {SVM_MODEL_PATH}")
        print(f"- {XGB_MODEL_PATH}")
        print(f"- {LE_PATH}")
        return

    # 2. Start the interactive prediction loop
    print("\nEnter a sentence to recognize its intent. Type 'exit' to close.")
    while True:
        sentence = input("\nYour sentence: ")
        if sentence.lower() == 'exit':
            print("Goodbye! 👋")
            break
        if not sentence.strip():
            print("Please enter a valid sentence.")
            continue

        # Generate embedding for the input sentence
        embedding = get_bert_embedding(sentence, bert_model, tokenizer)

        # --- SVM Prediction ---
        pred_svm_idx = model_svm.predict(embedding)[0]
        pred_svm_label = label_encoder.inverse_transform([pred_svm_idx])[0]
        # Get confidence scores (probabilities)
        # For SVM, we need to enable probability=True during training for this, 
        # but we can use decision_function as a proxy for confidence.
        # For simplicity, we'll just show the prediction. If probabilities are needed,
        # the SVM model needs to be retrained with probability=True.
        # For now, we get probabilities if available.
        try:
            confidence_svm = np.max(model_svm.predict_proba(embedding)) * 100
            print(f"\n🤖 SVM Model Prediction:")
            print(f"   Intent:     {pred_svm_label}")
            print(f"   Confidence: {confidence_svm:.2f}%")
        except AttributeError:
             print(f"\n🤖 SVM Model Prediction:")
             print(f"   Intent:     {pred_svm_label}")
             print(f"   (Confidence scores not available for this SVM model)")


        # --- XGBoost Prediction ---
        pred_xgb_idx = model_xgb.predict(embedding)[0]
        pred_xgb_label = label_encoder.inverse_transform([pred_xgb_idx])[0]
        confidence_xgb = np.max(model_xgb.predict_proba(embedding)) * 100
        
        print(f"\n🚀 XGBoost Model Prediction:")
        print(f"   Intent:     {pred_xgb_label}")
        print(f"   Confidence: {confidence_xgb:.2f}%")
        print("-" * 30)


if __name__ == '__main__':
    run_intent_recognizer()
