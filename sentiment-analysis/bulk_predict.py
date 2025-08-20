import pandas as pd
import pickle
import sys
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the saved model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load CSV safely
if len(sys.argv) < 2:
    print("Usage: python bulk_predict.py <input_file.csv>")
    sys.exit(1)

input_file = sys.argv[1]

try:
    df = pd.read_csv(input_file, quotechar='"', encoding="utf-8", on_bad_lines="skip")
except Exception as e:
    print("Error reading CSV:", e)
    sys.exit(1)

# Clean column names
df.columns = df.columns.str.strip()
if "text" not in df.columns:
    print("ERROR: CSV must have a 'text' column")
    sys.exit(1)

# Predict sentiment
X_vec = vectorizer.transform(df["text"].astype(str))
df["predicted_sentiment"] = model.predict(X_vec)

# Save predictions with safe filename
base_output_file = input_file.replace(".csv", "_with_sentiment.csv")
output_file = base_output_file
counter = 1
while os.path.exists(output_file):
    try:
        with open(output_file, "a"):
            break
    except PermissionError:
        output_file = base_output_file.replace(".csv", f"_{counter}.csv")
        counter += 1

df.to_csv(output_file, index=False)
print(f"Sentiment analysis complete. Results saved in {output_file}")

# Print quick stats
print("\nSentiment Counts:")
print(df["predicted_sentiment"].value_counts())

# Sentiment distribution
counts = df["predicted_sentiment"].value_counts()

# Pie chart
plt.figure(figsize=(6,6))
counts.plot.pie(autopct='%1.1f%%', startangle=90, shadow=True)
plt.title("Sentiment Distribution")
plt.ylabel("")
plt.show()

# Bar chart with fixed colors
color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
plt.figure(figsize=(6,4))
counts.plot.bar(color=[color_map.get(s, "blue") for s in counts.index])
plt.title("Sentiment Counts")
plt.ylabel("Number of Posts")
plt.show()

# WordClouds
for sentiment in df["predicted_sentiment"].unique():
    text_data = " ".join(df[df["predicted_sentiment"] == sentiment]["text"].astype(str))
    if text_data.strip():  # avoid error if empty
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Top Words in {sentiment.capitalize()} Posts")
        plt.show()
