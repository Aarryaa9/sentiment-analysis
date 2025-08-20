import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()  # clean column names
print("Columns in CSV:", df.columns.tolist())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Vectorizer (TF-IDF is more powerful than CountVectorizer)
vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models to try
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

best_model = None
best_acc = 0

# Train + evaluate each model
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    print(f" {name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, preds))

    # Track best model
    if acc > best_acc:
        best_acc = acc
        best_model = model

# Save best model + vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\nBest model saved: {type(best_model).__name__} with accuracy {best_acc:.2f}")
