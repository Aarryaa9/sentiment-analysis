import pickle

# Load the saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Ask user for input
while True:
    text = input("\nEnter a sentence to analyze sentiment (or type 'exit' to quit): ")
    if text.lower() == "exit":
        break

    # Convert text to features
    text_vec = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(text_vec)[0]
    print(f" Sentiment: {prediction}")
