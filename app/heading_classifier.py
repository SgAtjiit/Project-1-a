import joblib
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# Make sure to run 'python -m nltk.downloader punkt stopwords' once locally to download NLTK data

class HeadingClassifier:
    def __init__(self, model_dir="app/models/"):
        self.model_dir = model_dir
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Loads the pre-trained model and vectorizer."""
        model_path = os.path.join(self.model_dir, "heading_model.joblib")
        vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            # Fallback for hackathon: If model not found, use a simple heuristic.
            # In a real scenario, model training would be a separate step.
            print("Warning: Pre-trained NLP model not found. Using heuristic-based classification.")
            self.pipeline = None # Indicate that model is not loaded
            return

        try:
            self.pipeline = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            # Reconstruct the pipeline if necessary (e.g., if loaded separately)
            if not isinstance(self.pipeline, Pipeline):
                self.pipeline = Pipeline([
                    ('tfidf', self.vectorizer),
                    ('classifier', self.pipeline) # Assuming the loaded model is just the classifier
                ])
            print("NLP Model loaded successfully.")
        except Exception as e:
            print(f"Error loading NLP model: {e}. Falling back to heuristic-based classification.")
            self.pipeline = None

    def preprocess_text(self, text):
        """Applies basic text preprocessing."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return " ".join(tokens)

    def classify_heading(self, text_block):
        """
        Classifies a text block as H1, H2, H3, or Body.
        Prioritizes NLP model if available, falls back to heuristics.
        """
        text = text_block["text"]
        font_size = text_block["font_size"]
        page_num = text_block["page"]

        if self.pipeline:
            processed_text = self.preprocess_text(text)
            try:
                # Predict probability for each class if available
                probabilities = self.pipeline.predict_proba([processed_text])[0]
                classes = self.pipeline.classes_
                # Get the class with the highest probability
                predicted_class = classes[probabilities.argmax()]

                # Combine with heuristics for robustness, especially for edge cases
                # Example: if model predicts 'Body' but font size is very large and bold,
                # it might still be a title/heading.
                if predicted_class == "Body" and font_size > 20 and page_num == 1:
                    return "Title"
                elif predicted_class == "Body" and font_size > 16 and "Bold" in text_block["font_name"]:
                    return "H1"
                elif predicted_class == "Body" and font_size > 14 and "Bold" in text_block["font_name"]:
                    return "H2"
                elif predicted_class == "Body" and font_size > 12 and "Bold" in text_block["font_name"]:
                    return "H3"
                return predicted_class
            except Exception as e:
                print(f"Error during NLP prediction: {e}. Using heuristic for '{text}'.")
                return self._heuristic_classify(text_block)
        else:
            return self._heuristic_classify(text_block)

    def _heuristic_classify(self, text_block):
        """
        Heuristic-based classification (fallback if NLP model is not used/available).
        This is less robust but provides a baseline.
        """
        text = text_block["text"]
        font_size = text_block["font_size"]
        page_num = text_block["page"]

        # Simple font size based heuristics (can be refined)
        # These thresholds are highly document-dependent and should be tuned.
        if page_num == 1 and font_size >= 24: # Largest text on first page as potential title
            return "Title"
        elif font_size >= 18:
            return "H1"
        elif font_size >= 14:
            return "H2"
        elif font_size >= 12:
            return "H3"
        return "Body"

# --- Model Training Script (Run this locally ONCE to train and save your model) ---
# This part is NOT part of the Docker container's runtime. It's for development.

def train_and_save_model(output_dir="app/models/"):
    """
    Trains a simple NLP model for heading classification and saves it.
    You need to create your own `training_data` based on sample PDFs.
    """
    # Dummy training data (REPLACE with actual labeled data from PDFs)
    # In a real scenario, you'd manually label text blocks from diverse PDFs.
    # Example: text, label (Title, H1, H2, H3, Body)
    training_data = [
        ("Understanding AI", "Title"),
        ("1. Introduction", "H1"),
        ("1.1 What is AI?", "H2"),
        ("1.1.1 History of AI", "H3"),
        ("Artificial intelligence (AI) is a broad field of computer science.", "Body"),
        ("2. Machine Learning Fundamentals", "H1"),
        ("2.1 Supervised Learning", "H2"),
        ("2.1.1 Regression", "H3"),
        ("This section discusses the basics of machine learning algorithms.", "Body"),
        ("Chapter 3: Deep Learning", "H1"),
        ("Key Concepts in Neural Networks", "H2"),
        ("Activation Functions", "H3"),
        ("Deep learning is a subset of machine learning based on artificial neural networks.", "Body"),
        ("Conclusion and Future Work", "H1"),
        ("Summary", "H2"),
        ("Future Directions", "H3"),
        ("This document provides an overview of AI.", "Body"),
        ("APPENDIX A: Data Sources", "H1"),
        ("Table of Contents", "Body"), # Example of non-heading large text
        ("References", "H1"),
        ("Figure 1: AI Taxonomy", "Body")
    ]

    # Add some font size info to training data to make it more realistic for the model
    # This is a simplification; ideally, you'd extract font sizes during data collection.
    enriched_training_data = []
    for text, label in training_data:
        # Simulate font sizes based on typical heading structures
        font_size = 12 # Default for Body
        if label == "Title":
            font_size = 28
        elif label == "H1":
            font_size = 20
        elif label == "H2":
            font_size = 16
        elif label == "H3":
            font_size = 14
        enriched_training_data.append({"text": text, "label": label, "font_size": font_size})


    texts = [d["text"] for d in enriched_training_data]
    labels = [d["label"] for d in enriched_training_data]

    # Preprocess texts for training
    classifier_instance = HeadingClassifier(model_dir=output_dir) # Use instance to access preprocess_text
    processed_texts = [classifier_instance.preprocess_text(text) for text in texts]

    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))), # Limit features to keep model small
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train the model
    pipeline.fit(processed_texts, labels)

    # Save the model and vectorizer
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_dir, "heading_model.joblib"))
    joblib.dump(pipeline.named_steps['tfidf'], os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    print(f"Model and vectorizer saved to {output_dir}")

    # You can also evaluate the model here if you have a test set
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(processed_texts, labels, test_size=0.2, random_state=42)
    # print("Model accuracy:", pipeline.score(X_test, y_test))


# To train the model locally, uncomment the following line and run this file:
if __name__ == "__main__":
   train_and_save_model()