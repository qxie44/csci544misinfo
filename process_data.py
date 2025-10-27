import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_data(filepath: str) -> pd.DataFrame:
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
        'state', 'party', 'barely_true_ct', 'false_ct', 'half_true_ct', 
        'mostly_true_ct', 'pants_fire_ct', 'context'
    ]
    df = pd.read_csv(filepath, sep='\t', header=None, names=columns)
    count_cols = ['barely_true_ct', 'false_ct', 'half_true_ct', 'mostly_true_ct', 'pants_fire_ct']
    df[count_cols] = df[count_cols].apply(pd.to_numeric, errors='coerce')
    return df


def train_model(train_df, valid_df):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    X_train = vectorizer.fit_transform(train_df['statement'])
    y_train = train_df['label']
    
    X_valid = vectorizer.transform(valid_df['statement'])
    y_valid = valid_df['label']
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    valid_acc = accuracy_score(y_valid, model.predict(X_valid))
    
    print(f"Train accuracy: {train_acc:.1%}")
    print(f"Valid accuracy: {valid_acc:.1%}")
    
    return model, vectorizer


def evaluate(model, vectorizer, test_df):
    X_test = vectorizer.transform(test_df['statement'])
    y_test = test_df['label']
    y_pred = model.predict(X_test)
    
    test_df['predicted'] = y_pred
    test_df['correct'] = y_pred == y_test
    
    print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.1%}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return test_df


if __name__ == "__main__":
    print("Loading datasets...")
    train_df = load_data('train.tsv')
    valid_df = load_data('valid.tsv')
    test_df = load_data('test.tsv')
    
    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")
    print(f"\nLabel distribution:\n{train_df['label'].value_counts()}\n")
    
    print("Training model...")
    model, vectorizer = train_model(train_df, valid_df)
    
    print("\nEvaluating on test set...")
    test_df = evaluate(model, vectorizer, test_df)