# src/dashboard_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# ------------------
# Page settings
# ------------------
st.set_page_config(page_title="Healthcare VoC Analytics", layout="wide")
st.title("📊 Healthcare Voice of Customer Analytics Dashboard")
st.markdown("""
This dashboard provides **insights from patient feedback** across departments and visit types.
It includes **sentiment analysis, keyword trends, and topic modeling** to help improve patient experience.
""")

# ------------------
# Load dataset
# ------------------
@st.cache_data
def load_data():
    df = pd.read_csv('data/synthetic_feedback_cleaned.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    return df

df = load_data()

# ------------------
# Filters
# ------------------
departments = ['All'] + sorted(df['department'].unique())
visit_types = ['All'] + sorted(df['visit_type'].unique())

selected_dept = st.sidebar.selectbox("Department", departments)
selected_visit = st.sidebar.selectbox("Visit Type", visit_types)

df_filtered = df.copy()
if selected_dept != 'All':
    df_filtered = df_filtered[df_filtered['department'] == selected_dept]
if selected_visit != 'All':
    df_filtered = df_filtered[df_filtered['visit_type'] == selected_visit]

# ------------------
# NLP Sentiment Analysis
# ------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df_filtered['sentiment'] = df_filtered['clean_text'].apply(analyze_sentiment)

# ------------------
# Text preprocessing for topic modeling
# ------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df_filtered['text_clean_nlp'] = df_filtered['clean_text'].apply(clean_text)

# ------------------
# Overview metrics
# ------------------
st.header("🔹 Feedback Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Feedback", df_filtered.shape[0])
col2.metric("Positive Feedback", df_filtered[df_filtered['sentiment']=='Positive'].shape[0])
col3.metric("Neutral Feedback", df_filtered[df_filtered['sentiment']=='Neutral'].shape[0])
col4.metric("Negative Feedback", df_filtered[df_filtered['sentiment']=='Negative'].shape[0])
st.markdown("---")

# ------------------
# Sentiment Distribution
# ------------------
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots(figsize=(6,3))
sns.countplot(data=df_filtered, x='sentiment', order=['Positive','Neutral','Negative'], palette='Set2', ax=ax)
ax.set_ylabel("Count")
ax.set_title("Feedback Sentiment Count")
plt.tight_layout()
st.pyplot(fig)

# ------------------
# Average Rating by Department
# ------------------
st.subheader("Average Rating by Department")
avg_rating_dept = df_filtered.groupby('department')['rating'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6,3))
sns.barplot(x=avg_rating_dept.index, y=avg_rating_dept.values, palette='coolwarm', ax=ax)
ax.set_ylim(0,5)
ax.set_ylabel("Avg Rating")
ax.set_title("Average Rating by Department")
plt.tight_layout()
st.pyplot(fig)

# ------------------
# Sentiment Trend Over Time
# ------------------
st.subheader("Sentiment Trend Over Time")
monthly_sentiment = df_filtered.groupby(['month','sentiment']).size().unstack().fillna(0)
fig, ax = plt.subplots(figsize=(7,3))
monthly_sentiment.plot(kind='line', marker='o', ax=ax)
ax.set_ylabel("Number of Feedbacks")
ax.set_xlabel("Month")
ax.set_title("Monthly Feedback Sentiment Trend")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ------------------
# Word Clouds
# ------------------
st.subheader("Top Feedback Themes")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Negative Feedback Word Cloud**")
    text_neg = " ".join(df_filtered[df_filtered['sentiment']=='Negative']['text_clean_nlp'])
    if len(text_neg.strip())>0:
        wc_neg = WordCloud(width=400, height=200, background_color='white').generate(text_neg)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(wc_neg, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("No negative feedback in selection")

with col2:
    st.markdown("**Positive Feedback Word Cloud**")
    text_pos = " ".join(df_filtered[df_filtered['sentiment']=='Positive']['text_clean_nlp'])
    if len(text_pos.strip())>0:
        wc_pos = WordCloud(width=400, height=200, background_color='white').generate(text_pos)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(wc_pos, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("No positive feedback in selection")

# ------------------
# ------------------
# Topic Modeling (LDA) for Negative Feedback
# ------------------
st.subheader("Top Topics in Negative Feedback")

try:
    # Prepare text
    vectorizer = CountVectorizer(max_df=0.9, min_df=2)
    negative_texts = df_filtered[df_filtered['sentiment']=='Negative']['text_clean_nlp']
    
    if len(negative_texts) > 0:
        X = vectorizer.fit_transform(negative_texts)
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(X)
        terms = vectorizer.get_feature_names_out()

        # For each topic, get top words and feedback count
        for i, topic in enumerate(lda.components_):
            topic_terms = [terms[j] for j in topic.argsort()[-5:][::-1]]
            
            # Approximate feedback count per topic by assigning each document to the topic with highest weight
            doc_topic = lda.transform(X)
            topic_count = sum(doc_topic.argmax(axis=1) == i)
            
            # Assign human-readable topic labels (you can adjust based on word inspection)
            if i == 0:
                label = "Service Quality Issues / Confusing Instructions"
            elif i == 1:
                label = "Mixed Feedback / Average Service"
            else:
                label = "General Complaints / Average or Poor Service"
            
            st.markdown(f"💢 **Topic {i+1}: {label}**")
            st.markdown(f"**Top words:** {', '.join(topic_terms)}")
            st.markdown(f"**Feedback count:** {topic_count}")
            st.markdown("---")
    else:
        st.write("No negative feedback available for topic modeling.")

except Exception as e:
    st.write("Error in topic modeling:", e)

# ------------------
# Key Insights
# ------------------
st.header("💡 Key Insights")
st.markdown(f"""
- Filtered Department: **{selected_dept}**, Visit Type: **{selected_visit}**  
- Total Feedback: **{df_filtered.shape[0]}**  
- Positive: **{df_filtered[df_filtered['sentiment']=='Positive'].shape[0]}**, Negative: **{df_filtered[df_filtered['sentiment']=='Negative'].shape[0]}**  

- **Most common complaints (top words)**: {', '.join(text_neg.split()[:10]) if len(text_neg)>0 else 'N/A'}  
- **Most praised aspects (top words)**: {', '.join(text_pos.split()[:10]) if len(text_pos)>0 else 'N/A'}  

> Topic modeling shows the main themes in negative feedback to help stakeholders prioritize improvements.
""")
