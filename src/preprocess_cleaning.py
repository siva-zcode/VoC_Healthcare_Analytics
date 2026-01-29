# src/preprocess_cleaning.py
import pandas as pd
import string
from textblob import TextBlob

# Load dataset
df = pd.read_csv('../data/synthetic_feedback.csv')

# Remove missing feedback
df = df.dropna(subset=['feedback_text'])

# Clean text
df['clean_text'] = df['feedback_text'].str.lower().str.translate(str.maketrans('', '', string.punctuation))

# Sentiment Analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_text'].apply(get_sentiment)

# Save cleaned dataset
df.to_csv('../data/synthetic_feedback_cleaned.csv', index=False)
print("Data preprocessing & sentiment analysis done!")
