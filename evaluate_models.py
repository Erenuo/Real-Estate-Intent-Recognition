from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import torch
import nltk
import os

TEST_DATA_PATH = 'test.csv'
MODELS_DIR = 'saved_models'
RESULTS_DIR = 'evaluation_results'

# Helper Function for Encoders
def get_bert_embeddings(texts, model, tokenizer):
    """Generates BERT embeddings for a list of texts."""
    model.eval() # Put model in evaluation mode
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of the last hidden state as the sentence embedding
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)


def evaluate_all_models():
    """
    Loads all trained models and evaluates them against the test dataset.
    """
    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created directory: {RESULTS_DIR}")

    # Load test data
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        print(f"\nLoaded test data from '{TEST_DATA_PATH}' ({len(test_df)} rows)")
    except FileNotFoundError:
        print(f"Error: Test data not found at '{TEST_DATA_PATH}'")
        print("Please run the 'split_dataset.py' script first.")
        return

    # Find all model files and load the single label encoder
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.joblib')]
    if not model_files:
        print(f"Error: No models found in '{MODELS_DIR}' directory.")
        return
        
    print(f"Found {len(model_files)} models to evaluate.")

    try:
        # Load the single, shared label encoder
        le_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
        le = joblib.load(le_path)
        print(f"Successfully loaded shared label encoder from '{le_path}'")
    except FileNotFoundError:
        print(f"Error: The shared 'label_encoder.joblib' was not found in '{MODELS_DIR}'.")
        print("Please ensure at least one training script has been run to create it.")
        return

    # Pre-load BERT model and tokenizer to avoid loading them in the loop
    print("\nLoading BERT model (this may take a moment)...")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    nltk.download('punkt', quiet=True)


    # Loop through and evaluate each model
    for model_file in sorted(model_files):
        model_name = model_file.replace('_model.joblib', '')
        print(f"\n--- Evaluating Model: {model_name} ---")

        try:
            # Load the model
            model = joblib.load(os.path.join(MODELS_DIR, model_file))

            # Encode the true labels from the test set using the shared encoder
            y_true = le.transform(test_df['intent'])
            X_test_text = test_df['utterance']
            
            # Preprocess test data based on model type
            if 'char_tfidf' in model_name:
                vectorizer_path = os.path.join(MODELS_DIR, 'char_tfidf_vectorizer.joblib')
                vectorizer = joblib.load(vectorizer_path)
                X_test = vectorizer.transform(X_test_text)
            
            elif 'tfidf' in model_name:
                vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
                vectorizer = joblib.load(vectorizer_path)
                X_test = vectorizer.transform(X_test_text)

            elif 'bert' in model_name:
                X_test = get_bert_embeddings(X_test_text.tolist(), bert_model, bert_tokenizer)

            else:
                print(f"Warning: Unknown model type for {model_name}. Skipping.")
                continue

            # Make predictions
            y_pred = model.predict(X_test)

            # Generate and print classification report
            print("Classification Report:")
            report = classification_report(y_true, y_pred, target_names=le.classes_)
            print(report)

            # Generate and save confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Actual Intent')
            plt.xlabel('Predicted Intent')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            cm_path = os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")

        except FileNotFoundError as e:
            print(f"Error loading files for {model_name}: {e}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while evaluating {model_name}: {e}. Skipping.")


if __name__ == '__main__':
    evaluate_all_models()
