from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/sms_spam.csv"))
    parser.add_argument("--representation", choices=["bow", "tfidf"], default="tfidf")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    text_col = "Message" if "Message" in df.columns else "message"
    label_col = "Class" if "Class" in df.columns else "label"

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.25, random_state=42, stratify=df[label_col]
    )

    vectorizer = CountVectorizer() if args.representation == "bow" else TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = SVC(probability=True)
    model.fit(X_train_vec, y_train)
    print(classification_report(y_test, model.predict(X_test_vec)))


if __name__ == "__main__":
    main()
