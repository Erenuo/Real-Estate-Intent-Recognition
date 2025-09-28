from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import optuna
import torch
import os

# Helper Function for BERT
def get_bert_embeddings(texts, model, tokenizer):
    """Generates BERT embeddings for a list of texts."""
    model.eval() # Put model in evaluation mode
    embeddings = []
    print("Generating BERT embeddings...")
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of the last hidden state as the sentence embedding
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# Load and Prepare Data
# Using the original full dataset file
df = pd.read_csv('real_estate_intent_dataset.csv')
le = LabelEncoder()
df['intent_encoded'] = le.fit_transform(df['intent'])

# BERT Model and Tokenizer
print("Loading pre-trained BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Generate embeddings for the entire dataset
X_bert = get_bert_embeddings(df['utterance'].tolist(), bert_model, tokenizer)
y = df['intent_encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter Tuning with Optuna for XGBoost
def objective(trial):
    """Defines the search space for XGBoost hyperparameters."""
    
    param = {
        'objective': 'multi:softmax',
        'num_class': len(le.classes_),
        'eval_metric': 'mlogloss',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'use_label_encoder': False
    }

    model = xgb.XGBClassifier(**param, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return accuracy

print("\nStarting hyperparameter tuning with Optuna for XGBoost...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Train and Save Final Model
print(f"\nBest trial accuracy for XGBoost with BERT: {study.best_value:.4f}")
print("Best hyperparameters found: ", study.best_params)

# Create the final model with the best parameters
final_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    use_label_encoder=False,
    random_state=42,
    **study.best_params
)

# Train the final model on the training data
final_model.fit(X_train, y_train)

# Save the Model and Encoders
OUTPUT_DIR = 'saved_models'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'bert_xgboost_model.joblib'))
joblib.dump(le, os.path.join(OUTPUT_DIR, 'label_encoder.joblib'))

print("\nXGBoost model and label encoder saved successfully.")
