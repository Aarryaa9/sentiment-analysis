# Sentiment Analysis Project

This is a simple Python project that predicts the sentiment of text using **Naive Bayes** and **Bag-of-Words**.

---

## Project Overview

The project can:

- Train a sentiment analysis model on your dataset.
- Predict sentiments for new text data.
- Generate a chart showing how well the model performs.

---

## Project Structure

sentiment-analysis/
│
├── data.csv # Your training data
├── real_test.csv # New data to predict sentiments
├── train.py # Script to train the model
├── bulk_predict.py # Script to make predictions
├── requirements.txt # Needed Python packages
├── README.md
├── .gitignore
└── visuals/ # Generated chart
└── chart.png



## How to Use

1. Clone the repo:

```bash
git clone https://github.com/Aarryaa9/sentiment-analysis.git
cd sentiment-analysis
Install dependencies:

bash

pip install -r requirements.txt
Train the model:

bash

python train.py
This will train the model and save a performance chart as visuals/chart.png.

Predict new data:

bash

python bulk_predict.py real_test.csv
This will give you predicted sentiments for your test data.

Visualization
Chart: visuals/chart.png — shows the model’s performance in a simple visual way.

License
This project is open for anyone to use and learn from.




