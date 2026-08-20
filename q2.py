"""
COIT29225 Neural Networks & Deep Learning
Assignment 1 - Problem 2

BBC News Text Classification

This program will:
- Load and inspect the supplied BBC News dataset.
- Validate the dataset before text preprocessing.
- Preprocess news article text.
- Implement TF-IDF + MLP classification.
- Implement Word2Vec + MLP classification.
- Evaluate and compare both classification approaches.
"""

import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Path to the supplied BBC News dataset.
DATA_FILE = "data/BBCNews.csv"

CATEGORY_NAMES = {
    "business": "Business",
    "entertainment": "Entertainment",
    "politics": "Politics",
    "sport": "Sport",
    "tech": "Technology",
}


def load_data(file_path):
    """
    Load the BBC News dataset from a CSV file.

    Args:
        file_path (str): Path to the BBC News CSV file.

    Returns:
        pandas.DataFrame: Loaded BBC News dataset.
    """
    data = pd.read_csv(file_path)

    return data


def inspect_data(data):
    """
    Inspect the structure and quality of the BBC News dataset.

    Args:
        data (pandas.DataFrame): BBC News dataset to inspect.
    """
    print("\nFirst five records:")
    print(data.head())

    print("\nDataset dimensions:")
    print(data.shape)

    print("\nColumn names:")
    print(data.columns)

    print("\nData types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:")
    print(data.duplicated().sum())

    print("\nCategory distribution:")
    print(data["category"].value_counts())


def clean_data(data):
    """
    Clean the BBC News dataset before text preprocessing.

    The function:
    - Removes exact duplicate rows.
    - Normalises the category labels to the five assignment categories.

    Args:
        data (pandas.DataFrame): Original BBC News dataset.

    Returns:
        pandas.DataFrame: Cleaned BBC News dataset.
    """

    # Create a copy so the original DataFrame is not modified directly.
    cleaned_data = data.copy()

    # Remove exact duplicate articles before the train/test split.
    cleaned_data = cleaned_data.drop_duplicates().reset_index(drop=True)

    # Convert dataset category labels to the names used in the assignment.
    cleaned_data["category"] = cleaned_data["category"].map(CATEGORY_NAMES)

    return cleaned_data


def display_cleaning_results(original_data, cleaned_data):
    """
    Display the results of the dataset cleaning process.

    Args:
        original_data (pandas.DataFrame): Dataset before cleaning.
        cleaned_data (pandas.DataFrame): Dataset after cleaning.
    """

    removed_duplicates = len(original_data) - len(cleaned_data)

    print("\nDataset Cleaning:")
    print(f"Original rows: {len(original_data)}")
    print(f"Duplicate rows removed: {removed_duplicates}")
    print(f"Rows after cleaning: {len(cleaned_data)}")

    print("\nCleaned category distribution:")
    print(cleaned_data["category"].value_counts())


def preprocess_text(text):
    """
    Preprocess one BBC News article.

    The preprocessing steps include:
    - Converting text to lowercase.
    - Removing punctuation and special characters.
    - Tokenising the text using whitespace.
    - Removing English stop words.

    Args:
        text (str): Original news article text.

    Returns:
        list: Preprocessed tokens from the article.
    """

    # Convert the complete article to lowercase.
    text = text.lower()

    # Replace punctuation and special characters with spaces.
    # Letters, numbers, and whitespace are retained.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Split the cleaned text into individual word tokens.
    tokens = text.split()

    # Remove common English stop words such as "the", "and", and "is".
    tokens = [
        token
        for token in tokens
        if token not in ENGLISH_STOP_WORDS
    ]

    return tokens 


def preprocess_data(data):
    """
    Preprocess the text of every article in the BBC News dataset.

    Two new columns are created:
    - tokens: tokenised text used by Word2Vec.
    - processed_text: cleaned text used by TF-IDF.

    Args:
        data (pandas.DataFrame): Cleaned BBC News dataset.

    Returns:
        pandas.DataFrame: Dataset containing preprocessed text.
    """

    processed_data = data.copy()

    # Apply the preprocessing function to every news article.
    processed_data["tokens"] = (
        processed_data["text"]
        .apply(preprocess_text)
    )

    # Join the token lists back into strings for TF-IDF.
    processed_data["processed_text"] = (
        processed_data["tokens"]
        .apply(" ".join)
    )

    return processed_data


def display_preprocessing_results(data):
    """
    Display examples and statistics from text preprocessing.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.
    """

    print("\nText Preprocessing:")

    print("\nOriginal text example:")
    print(data.loc[0, "text"][:300])

    print("\nProcessed text example:")
    print(data.loc[0, "processed_text"][:300])

    empty_documents = (
        data["processed_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    token_counts = data["tokens"].apply(len)

    print("\nPreprocessing statistics:")
    print(f"Empty processed documents: {empty_documents}")
    print(
        f"Average tokens per article: "
        f"{token_counts.mean():.2f}"
    )
    print(
        f"Minimum tokens in an article: "
        f"{token_counts.min()}"
    )
    print(
        f"Maximum tokens in an article: "
        f"{token_counts.max()}"
    )

def main():
    """
    Run the BBC News Text Classification application.
    """
    data = load_data(DATA_FILE)

    inspect_data(data)

    cleaned_data = clean_data(data)

    display_cleaning_results(
        data,
        cleaned_data,
    )

    processed_data = preprocess_data(cleaned_data)

    display_preprocessing_results(processed_data)

if __name__ == "__main__":
    main()