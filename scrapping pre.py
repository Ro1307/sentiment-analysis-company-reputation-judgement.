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
import time

# Ensure nltk components are downloaded
nltk.download('vader_lexicon')

# Function to scrape reviews from Trustpilot using Selenium
def scrape_reviews(company_name):
    company_name = company_name.replace("www.", "").strip()
    url = f"https://www.trustpilot.com/review/{company_name}"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.XPATH, "//p[contains(@class, 'typography_body')]")))
    except:
        print("No reviews found or page did not load correctly.")
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
    except Exception as e:
        print("Error extracting reviews:", e)

    driver.quit()
    print("Extracted Reviews:", reviews)
    return reviews

# Function to analyze sentiment
def analyze_sentiment(reviews):
    sia = SentimentIntensityAnalyzer()
    sentiments = []

    for review in reviews:
        score = sia.polarity_scores(review)
        if -0.05 < score['compound'] < 0.05:
            continue
        sentiments.append(score['compound'])

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    if avg_sentiment > 0.3:
        reputation = "Good"
    elif avg_sentiment < -0.3:
        reputation = "Bad"
    else:
        reputation = "Neutral"

    print(f"\nOverall Sentiment Score: {avg_sentiment:.4f}")
    print(f"Company Reputation: {reputation}")

    return sentiments, reputation

# Function to visualize sentiment histogram
def plot_sentiment(sentiments, company_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiments, bins=20, kde=True, color='skyblue')

    avg_sent = np.mean(sentiments)
    plt.axvline(avg_sent, color='red', linestyle='--', label=f'Avg Sentiment: {avg_sent:.2f}')

    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.title(f"Sentiment Distribution for {company_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot bar chart for sentiment categories
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
    plt.show()

company = input("Enter the Trustpilot company domain (e.g., 'amazon.com'): ").strip().lower()
reviews = scrape_reviews(company)

if not reviews:
    print("No meaningful reviews found. Try another company.")
else:
    sentiments, reputation = analyze_sentiment(reviews)
    plot_sentiment(sentiments, company)
    plot_sentiment_categories(sentiments)