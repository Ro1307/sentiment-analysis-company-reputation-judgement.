from flask import Flask, request, render_template_string
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure nltk components are downloaded
nltk.download('vader_lexicon')

# Function to scrape reviews from Trustpilot using Selenium
def scrape_reviews(company_name):
    company_name = company_name.replace("www.", "").strip()
    url = f"https://www.trustpilot.com/review/{company_name}"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, "//p[contains(@class, 'typography_body')]")))
    except:
        driver.quit()
        return []

    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    reviews = []
    try:
        review_elements = driver.find_elements(By.XPATH, "//p[contains(@class, 'typography_body')]")
        for review in review_elements:
            cleaned_review = review.text.strip()
            if len(cleaned_review.split()) > 3:
                reviews.append(cleaned_review)
    except:
        pass

    driver.quit()
    return reviews

# Function to analyze sentiment
def analyze_sentiment(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(review)['compound'] for review in reviews]
    filtered = [s for s in sentiments if s >= 0.05 or s <= -0.05]
    avg_sentiment = sum(filtered) / len(filtered) if filtered else 0

    if avg_sentiment > 0.3:
        reputation = "Good"
    elif avg_sentiment < -0.3:
        reputation = "Bad"
    else:
        reputation = "Neutral"

    return sentiments, reputation, avg_sentiment

# Visualization functions
def plot_sentiment(sentiments, company_name):
    categories = ['Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral' for s in sentiments]
    df = pd.DataFrame({'Sentiment': sentiments, 'Category': categories})
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Sentiment', bins=20, hue='Category',
                 palette={'Positive': 'green', 'Neutral': 'grey', 'Negative': 'red'}, multiple='stack')
    plt.axvline(np.mean(sentiments), color='blue', linestyle='--', label=f'Avg Sentiment: {np.mean(sentiments):.2f}')
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.title(f"Sentiment Distribution for {company_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/hist.png')
    plt.close()

def plot_sentiment_categories(sentiments):
    positive = sum(1 for s in sentiments if s > 0.05)
    negative = sum(1 for s in sentiments if s < -0.05)
    neutral = len(sentiments) - positive - negative
    labels = ['Positive', 'Neutral', 'Negative']
    values = [positive, neutral, negative]
    colors = ['green', 'grey', 'red']
    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=values, palette=colors)
    plt.title("Sentiment Category Counts")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig('static/bar.png')
    plt.close()

def plot_sentiment_pie(sentiments):
    positive = sum(1 for s in sentiments if s > 0.05)
    negative = sum(1 for s in sentiments if s < -0.05)
    neutral = len(sentiments) - positive - negative
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [positive, neutral, negative]
    colors = ['green', 'grey', 'red']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title("Sentiment Category Distribution (Pie Chart)")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('static/pie.png')
    plt.close()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analyzer</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 30px; text-align: center; }
        input { padding: 10px; font-size: 16px; width: 300px; }
        button { padding: 10px 20px; font-size: 16px; margin-top: 10px; }
        img { margin: 20px auto; width: 80%; max-width: 600px; }
    </style>
</head>
<body>
    {% if not results %}
        <h2>Enter Trustpilot Company Domain</h2>
        <form action="/" method="post">
            <input name="company" placeholder="e.g. amazon.com">
            <br>
            <button type="submit">Analyze</button>
        </form>
    {% else %}
        <h2>Sentiment Results for {{ company }}</h2>
        <p><strong>Reputation:</strong> {{ reputation }}</p>
        <p><strong>Average Sentiment Score:</strong> {{ score }}</p>
        <img src="/static/hist.png">
        <img src="/static/bar.png">
        <img src="/static/pie.png">
        <br><a href="/">Try another company</a>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        company = request.form['company']
        reviews = scrape_reviews(company)
        if not reviews:
            return render_template_string(HTML_TEMPLATE, results=False)
        sentiments, reputation, avg_score = analyze_sentiment(reviews)
        os.makedirs('static', exist_ok=True)
        plot_sentiment(sentiments, company)
        plot_sentiment_categories(sentiments)
        plot_sentiment_pie(sentiments)
        return render_template_string(HTML_TEMPLATE, results=True, company=company,
                                      reputation=reputation, score=f"{avg_score:.4f}")
    return render_template_string(HTML_TEMPLATE, results=False)

if __name__ == "__main__":
    app.run(debug=True)
