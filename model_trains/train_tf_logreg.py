import pandas as pd
import optuna
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- 1. Load and Prepare Data ---
df = pd.read_csv('train.csv')
le = LabelEncoder()
df['intent_encoded'] = le.fit_transform(df['intent'])
X = df['utterance']
y = df['intent_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 2. TF-IDF Vectorization ---
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# --- 3. Hyperparameter Tuning with Optuna ---
def objective(trial):
    solver = 'liblinear'
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    C = trial.suggest_float('C', 1e-2, 1e2, log=True)
    model = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return accuracy_score(y_test, model.predict(X_test_tfidf))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# --- 4. Train and Save Final Model ---
print(f"Best trial accuracy for LogReg with TF-IDF: {study.best_value:.4f}")
print("Best hyperparameters: ", study.best_params)

final_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42, **study.best_params)
final_model.fit(X_train_tfidf, y_train)

# --- 5. Save the Model and Encoders ---
joblib.dump(final_model, 'saved_models/tfidf_logreg_model.joblib')
joblib.dump(tfidf_vectorizer, 'saved_models/tfidf_vectorizer.joblib')
joblib.dump(le, 'saved_models/label_encoder.joblib')

print("\nModel, TF-IDF vectorizer, and label encoder saved successfully.")
