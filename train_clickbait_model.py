import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def main():
    print("Starting script...")

    data_path = "datasets/clickbait_datasets/train1.csv"  # fixed path string, no extra quotes

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    print(f"Dataset found at {data_path}")

    df = pd.read_csv(data_path)
    print("Dataset loaded")
    print("Columns in CSV:", df.columns.tolist())

    if 'headline' not in df.columns or 'clickbait' not in df.columns:
        raise ValueError("CSV must have 'headline' and 'clickbait' columns.")
    print("Required columns are present")

    X = df['headline']
    y = df['clickbait']

    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    print("Vectorization done")

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    print("Train-test split done")

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model training done")

    joblib.dump(model, "clickbait_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("✅ Model and vectorizer saved!")

if __name__ == "__main__":
    main()
