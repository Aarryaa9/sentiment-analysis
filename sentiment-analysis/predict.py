import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Sentiment Analyzer is ready. Type your sentences below.")
print("(Type 'exit' anytime to stop)\n")

while True:
    user_input = input("Enter a sentence: ")
    if user_input.strip().lower() == "exit":
        print("\nExiting... Goodbye!")
        break

    text_features = vectorizer.transform([user_input])
    predicted_sentiment = model.predict(text_features)[0]
    print(f"Sentiment detected: {predicted_sentiment}\n")
