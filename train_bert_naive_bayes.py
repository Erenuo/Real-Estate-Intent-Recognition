import pandas as pd
import numpy as np
import optuna
import joblib
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# --- Helper Function for BERT ---
def get_bert_embeddings(texts, model, tokenizer):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)

# --- 1. Load and Prepare Data ---
df = pd.read_csv('train.csv')
le = LabelEncoder()
df['intent_encoded'] = le.fit_transform(df['intent'])

# --- 2. BERT Model and Tokenizer ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
X_bert = get_bert_embeddings(df['utterance'].tolist(), bert_model, tokenizer)
y = df['intent_encoded']
X_train, X_test, y_train, y_test = train_test_split(X_bert, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Hyperparameter Tuning with Optuna ---
def objective(trial):
    var_smoothing = trial.suggest_float('var_smoothing', 1e-10, 1e-8, log=True)
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# --- 4. Train and Save Final Model ---
print(f"Best trial accuracy for Naive Bayes with BERT: {study.best_value:.4f}")
print("Best hyperparameters: ", study.best_params)

final_model = GaussianNB(**study.best_params)
final_model.fit(X_train, y_train)

# --- 5. Save the Model and Encoders ---
joblib.dump(final_model, 'saved_models/bert_naive_bayes_model.joblib')
joblib.dump(le, 'saved_models/label_encoder.joblib')

print("\nModel and label encoder saved successfully.")
