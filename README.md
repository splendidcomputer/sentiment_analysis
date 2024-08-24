# Sentiment Analysis on Movie Reviews

This repository contains a Python script to perform sentiment analysis on movie reviews using a Naive Bayes classifier. The dataset used for this analysis is the `movie_reviews` corpus from the Natural Language Toolkit (nltk). The script trains a model to classify movie reviews as either positive or negative based on the text of the reviews.

## Table of Contents

- [Sentiment Analysis on Movie Reviews](#sentiment-analysis-on-movie-reviews)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

Before running the script, you need to install the required Python libraries. You can install them using `pip`:

```bash
pip install nltk pandas scikit-learn
```

## Data Preparation

The script uses the `movie_reviews` corpus from the `nltk` library. If you haven't downloaded the `movie_reviews` dataset, the script will download it automatically.

```python
nltk.download("movie_reviews")

# Load the dataset
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Convert to DataFrame
df = pd.DataFrame(documents, columns=["review", "sentiment"])
```

## Model Training

The model is trained using a Naive Bayes classifier from `scikit-learn`. The text data is converted into feature vectors using `CountVectorizer`, which transforms the text data into a matrix of token counts.

```python
# Convert text data to feature vectors
vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(df["review"])
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)
```

## Evaluation

After training the model, its performance is evaluated using the test set. The script prints the accuracy and a detailed classification report.

```python
# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
```

## Prediction

The script also includes a function `predict_sentiment` to predict the sentiment of new movie reviews.

```python
def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

# Test the prediction function
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("The movie was okay, nothing special."))
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-movie-reviews.git
   cd sentiment-analysis-movie-reviews
   ```

2. **Install the required libraries:**

   ```bash
   pip install nltk pandas scikit-learn
   ```

3. **Run the script:**

   ```bash
   python sentiment_analysis.py
   ```

4. **Use the `predict_sentiment` function** to predict the sentiment of a new review.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
