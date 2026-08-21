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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    TfidfVectorizer,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
import numpy as np
from gensim.models import Word2Vec

# Path to the supplied BBC News dataset.
DATA_FILE = "data/BBCNews.csv"
RANDOM_STATE = 42
TFIDF_OUTPUT_DIR = "output/tfidf"
WORD2VEC_OUTPUT_DIR = "output/word2vec"
WORD2VEC_VECTOR_SIZE = 100

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

def split_dataset(data, feature_column):
    """
    Split the preprocessed BBC News dataset into training, validation,
    and testing subsets.

    The testing set contains 20% of the complete dataset. The remaining
    training portion is further divided so validation data can be used
    to monitor model accuracy during training.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.

    Returns:
        tuple: Training, validation, and testing text and labels.
    """

    texts = data[feature_column]
    labels = data["category"]

    # Keep 20% of the complete dataset separate for final testing.
    (
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    ) = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Split the remaining data again to create a validation set.
    (
        train_texts,
        validation_texts,
        train_labels,
        validation_labels,
    ) = train_test_split(
        train_texts,
        train_labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_labels,
    )

    return (
        train_texts,
        validation_texts,
        test_texts,
        train_labels,
        validation_labels,
        test_labels,
    )


def create_tfidf_features(
    train_texts,
    validation_texts,
    test_texts,
):
    """
    Convert preprocessed BBC News articles into TF-IDF feature vectors.

    The vectorizer is fitted only on the training data to prevent
    information from the validation or testing sets leaking into training.

    Args:
        train_texts (pandas.Series): Training article text.
        validation_texts (pandas.Series): Validation article text.
        test_texts (pandas.Series): Testing article text.

    Returns:
        tuple: TF-IDF vectorizer and transformed feature matrices.
    """

    vectorizer = TfidfVectorizer(
        max_features=5000,
    )

    train_features = vectorizer.fit_transform(train_texts)

    validation_features = vectorizer.transform(
        validation_texts
    )

    test_features = vectorizer.transform(
        test_texts
    )

    return (
        vectorizer,
        train_features,
        validation_features,
        test_features,
    )


def train_tfidf_mlp(
    train_features,
    validation_features,
    train_labels,
    validation_labels,
    epochs=30,
):
    """
    Train an MLP classifier using TF-IDF features.

    The model is trained one epoch at a time so training loss,
    training accuracy, and validation accuracy can be recorded.

    Args:
        train_features: TF-IDF training feature matrix.
        validation_features: TF-IDF validation feature matrix.
        train_labels (pandas.Series): Training labels.
        validation_labels (pandas.Series): Validation labels.
        epochs (int): Number of training epochs.

    Returns:
        tuple: Trained model and recorded training history.
    """

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        batch_size=32,
        learning_rate_init=0.001,
        random_state=RANDOM_STATE,
    )

    classes = sorted(train_labels.unique())

    loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    best_model = None
    best_validation_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.partial_fit(
            train_features,
            train_labels,
            classes=classes,
        )

        training_predictions = model.predict(
            train_features
        )

        validation_predictions = model.predict(
            validation_features
        )

        training_accuracy = accuracy_score(
            train_labels,
            training_predictions,
        )

        validation_accuracy = accuracy_score(
            validation_labels,
            validation_predictions,
        )

        loss_history.append(model.loss_)
        training_accuracy_history.append(
            training_accuracy
        )
        validation_accuracy_history.append(
            validation_accuracy
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            best_model = deepcopy(model)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} - "
            f"Loss: {model.loss_:.4f} - "
            f"Training Accuracy: {training_accuracy:.4f} - "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    history = {
        "loss": loss_history,
        "training_accuracy": training_accuracy_history,
        "validation_accuracy": validation_accuracy_history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
}

    return best_model, history


def display_tfidf_setup(
    train_labels,
    validation_labels,
    test_labels,
    vectorizer,
):
    """
    Display information about the TF-IDF training setup.
    """

    print("\nTF-IDF + MLP Dataset Split:")
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(validation_labels)}")
    print(f"Testing samples: {len(test_labels)}")

    print(
        f"TF-IDF vocabulary size: "
        f"{len(vectorizer.get_feature_names_out())}"
    )

def run_tfidf_mlp(data):
    """
    Run the TF-IDF + MLP classification pipeline.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.
    """

    (
        train_texts,
        validation_texts,
        test_texts,
        train_labels,
        validation_labels,
        test_labels,
    ) = split_dataset(
        data,
        "processed_text"
    )

    (
        vectorizer,
        train_features,
        validation_features,
        test_features,
    ) = create_tfidf_features(
        train_texts,
        validation_texts,
        test_texts,
    )

    display_tfidf_setup(
        train_labels,
        validation_labels,
        test_labels,
        vectorizer,
    )

    model, history = train_tfidf_mlp(
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    )

    print(
        "\nBest validation result:"
        f"\nEpoch: {history['best_epoch']}"
        f"\nValidation Accuracy: "
        f"{history['best_validation_accuracy']:.4f}"
    )

    save_tfidf_training_curves(history)

    predictions, metrics = evaluate_tfidf_model(
        model,
        test_features,
        test_labels,
    )

    display_random_predictions(
        model,
        vectorizer,
        test_texts,
        test_labels,
    )

    print("\nTF-IDF + MLP pipeline complete.")


def save_tfidf_training_curves(history):
    """
    Save the TF-IDF + MLP training curves required for Problem 2.

    Args:
        history (dict): Recorded loss and accuracy values by epoch.
    """

    Path(TFIDF_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history["loss"]) + 1,
    )

    # -----------------------------------------------------------------------
    # Training Loss Curve
    # -----------------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
    )

    plt.title("TF-IDF + MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{TFIDF_OUTPUT_DIR}/training_loss.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------------------------
    # Training and Validation Accuracy Curve
    # -----------------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["training_accuracy"],
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["validation_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )

    plt.title("TF-IDF + MLP Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{TFIDF_OUTPUT_DIR}/accuracy_curve.png",
        dpi=300,
    )

    plt.close()

    print(
        f"\nTF-IDF training curves saved to: "
        f"{TFIDF_OUTPUT_DIR}"
    )


def evaluate_tfidf_model(
    model,
    test_features,
    test_labels,
):
    """
    Evaluate the trained TF-IDF + MLP model on the final testing set.

    Args:
        model (MLPClassifier): Trained MLP classifier.
        test_features: TF-IDF testing feature matrix.
        test_labels (pandas.Series): True testing labels.

    Returns:
        tuple: Test predictions and calculated performance metrics.
    """

    predictions = model.predict(test_features)

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    (
        precision,
        recall,
        f1_score,
        _,
    ) = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nTF-IDF + MLP Test Results:")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            predictions,
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    return predictions, metrics


def display_random_predictions(
    model,
    vectorizer,
    test_texts,
    test_labels,
    sample_count=10,
):
    """
    Display predictions for random testing articles.

    Args:
        model (MLPClassifier): Trained classifier.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        test_texts (pandas.Series): Testing article text.
        test_labels (pandas.Series): True testing labels.
        sample_count (int): Number of random samples to display.
    """

    sample_indices = test_texts.sample(
        n=sample_count,
        random_state=RANDOM_STATE,
    ).index

    print("\n10 Random TF-IDF + MLP Predictions:")

    for number, index in enumerate(
        sample_indices,
        start=1,
    ):
        article_text = test_texts.loc[index]
        actual_category = test_labels.loc[index]

        article_features = vectorizer.transform(
            [article_text]
        )

        predicted_category = model.predict(
            article_features
        )[0]

        print(f"\nSample {number}")
        print(f"Actual:    {actual_category}")
        print(f"Predicted: {predicted_category}")

        print(
            "Text:      "
            f"{article_text[:150]}..."
        )


def create_document_vector(tokens, word2vec_model):
    """
    Create one article vector by averaging the Word2Vec vectors of
    recognised words in the article.

    Args:
        tokens (list): Preprocessed article tokens.
        word2vec_model (Word2Vec): Trained Word2Vec model.

    Returns:
        numpy.ndarray: Averaged article vector.
    """

    word_vectors = [
        word2vec_model.wv[token]
        for token in tokens
        if token in word2vec_model.wv
    ]

    if not word_vectors:
        return np.zeros(
            WORD2VEC_VECTOR_SIZE,
            dtype=float,
        )

    return np.mean(
        word_vectors,
        axis=0,
    )


def create_word2vec_features(
    train_tokens,
    validation_tokens,
    test_tokens,
):
    """
    Train Word2Vec on the training articles and convert each article
    into an averaged Word2Vec feature vector.

    Args:
        train_tokens (pandas.Series): Training token lists.
        validation_tokens (pandas.Series): Validation token lists.
        test_tokens (pandas.Series): Testing token lists.

    Returns:
        tuple: Trained Word2Vec model and document feature matrices.
    """

    word2vec_model = Word2Vec(
        sentences=train_tokens.tolist(),
        vector_size=WORD2VEC_VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=1,
        seed=RANDOM_STATE,
        epochs=20,
    )

    train_features = np.vstack([
        create_document_vector(
            tokens,
            word2vec_model,
        )
        for tokens in train_tokens
    ])

    validation_features = np.vstack([
        create_document_vector(
            tokens,
            word2vec_model,
        )
        for tokens in validation_tokens
    ])

    test_features = np.vstack([
        create_document_vector(
            tokens,
            word2vec_model,
        )
        for tokens in test_tokens
    ])

    return (
        word2vec_model,
        train_features,
        validation_features,
        test_features,
    )


def train_word2vec_mlp(
    train_features,
    validation_features,
    train_labels,
    validation_labels,
    epochs=30,
):
    """
    Train an MLP classifier using averaged Word2Vec article vectors.

    Args:
        train_features: Word2Vec training feature matrix.
        validation_features: Word2Vec validation feature matrix.
        train_labels (pandas.Series): Training labels.
        validation_labels (pandas.Series): Validation labels.
        epochs (int): Number of training epochs.

    Returns:
        tuple: Best trained model and training history.
    """

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        batch_size=32,
        learning_rate_init=0.001,
        random_state=RANDOM_STATE,
    )

    classes = sorted(train_labels.unique())

    loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    best_model = None
    best_validation_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.partial_fit(
            train_features,
            train_labels,
            classes=classes,
        )

        training_predictions = model.predict(
            train_features
        )

        validation_predictions = model.predict(
            validation_features
        )

        training_accuracy = accuracy_score(
            train_labels,
            training_predictions,
        )

        validation_accuracy = accuracy_score(
            validation_labels,
            validation_predictions,
        )

        loss_history.append(model.loss_)
        training_accuracy_history.append(
            training_accuracy
        )
        validation_accuracy_history.append(
            validation_accuracy
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch + 1
            best_model = deepcopy(model)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} - "
            f"Loss: {model.loss_:.4f} - "
            f"Training Accuracy: {training_accuracy:.4f} - "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    history = {
        "loss": loss_history,
        "training_accuracy": training_accuracy_history,
        "validation_accuracy": validation_accuracy_history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
    }

    return best_model, history


def run_word2vec_mlp(data):
    """
    Run the Word2Vec + MLP training pipeline.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.
    """

    (
        train_tokens,
        validation_tokens,
        test_tokens,
        train_labels,
        validation_labels,
        test_labels,
    ) = split_dataset(
        data,
        "tokens",
    )

    (
        word2vec_model,
        train_features,
        validation_features,
        test_features,
    ) = create_word2vec_features(
        train_tokens,
        validation_tokens,
        test_tokens,
    )

    print("\nWord2Vec + MLP Dataset Split:")
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(validation_labels)}")
    print(f"Testing samples: {len(test_labels)}")

    print(
        f"Word2Vec vocabulary size: "
        f"{len(word2vec_model.wv)}"
    )

    print(
        f"Word2Vec vector size: "
        f"{word2vec_model.vector_size}"
    )

    model, history = train_word2vec_mlp(
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    )

    print(
        "\nBest Word2Vec validation result:"
        f"\nEpoch: {history['best_epoch']}"
        f"\nValidation Accuracy: "
        f"{history['best_validation_accuracy']:.4f}"
    )

    save_word2vec_training_curves(history)

    predictions, metrics = evaluate_word2vec_model(
        model,
        test_features,
        test_labels,
    )

    display_random_word2vec_predictions(
        model,
        word2vec_model,
        test_tokens,
        test_labels,
    )

    print("\nWord2Vec + MLP pipeline complete.")

    return {
        "model": model,
        "word2vec_model": word2vec_model,
        "history": history,
        "metrics": metrics,
        "predictions": predictions,
    }

    print("\nWord2Vec + MLP training complete.")


def save_word2vec_training_curves(history):
    """
    Save the Word2Vec + MLP training curves required for Problem 2.

    Args:
        history (dict): Recorded loss and accuracy values by epoch.
    """

    Path(WORD2VEC_OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history["loss"]) + 1,
    )

    # -----------------------------------------------------------------------
    # Training Loss Curve
    # -----------------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
    )

    plt.title("Word2Vec + MLP Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{WORD2VEC_OUTPUT_DIR}/training_loss.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------------------------
    # Training and Validation Accuracy Curve
    # -----------------------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        epochs,
        history["training_accuracy"],
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        history["validation_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )

    plt.title("Word2Vec + MLP Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{WORD2VEC_OUTPUT_DIR}/accuracy_curve.png",
        dpi=300,
    )

    plt.close()

    print(
        f"\nWord2Vec training curves saved to: "
        f"{WORD2VEC_OUTPUT_DIR}"
    )


def evaluate_word2vec_model(
    model,
    test_features,
    test_labels,
):
    """
    Evaluate the Word2Vec + MLP model on the final testing set.

    Args:
        model (MLPClassifier): Trained Word2Vec MLP classifier.
        test_features: Word2Vec testing feature matrix.
        test_labels (pandas.Series): True testing labels.

    Returns:
        tuple: Test predictions and calculated performance metrics.
    """

    predictions = model.predict(test_features)

    accuracy = accuracy_score(
        test_labels,
        predictions,
    )

    (
        precision,
        recall,
        f1_score,
        _,
    ) = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nWord2Vec + MLP Test Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            predictions,
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

    return predictions, metrics


def display_random_word2vec_predictions(
    model,
    word2vec_model,
    test_tokens,
    test_labels,
    sample_count=10,
):
    """
    Display predictions for random Word2Vec testing articles.

    Args:
        model (MLPClassifier): Trained classifier.
        word2vec_model (Word2Vec): Trained Word2Vec model.
        test_tokens (pandas.Series): Testing article token lists.
        test_labels (pandas.Series): True testing labels.
        sample_count (int): Number of random samples to display.
    """

    sample_indices = test_tokens.sample(
        n=sample_count,
        random_state=RANDOM_STATE,
    ).index

    print("\n10 Random Word2Vec + MLP Predictions:")

    for number, index in enumerate(
        sample_indices,
        start=1,
    ):
        article_tokens = test_tokens.loc[index]
        actual_category = test_labels.loc[index]

        article_vector = create_document_vector(
            article_tokens,
            word2vec_model,
        ).reshape(1, -1)

        predicted_category = model.predict(
            article_vector
        )[0]

        article_preview = " ".join(
            article_tokens[:25]
        )

        print(f"\nSample {number}")
        print(f"Actual:    {actual_category}")
        print(f"Predicted: {predicted_category}")
        print(f"Text:      {article_preview}...")





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

    # Run the TF-IDF + MLP classification pipeline.
    run_tfidf_mlp(processed_data)

    # Run the Word2Vec + MLP classification pipeline.
    run_word2vec_mlp(processed_data)

if __name__ == "__main__":
    main()