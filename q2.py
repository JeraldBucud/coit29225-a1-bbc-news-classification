"""
COIT29225 Neural Networks & Deep Learning
Assignment 1 - Problem 2

BBC News Text Classification

This program will:
- Load and inspect the supplied BBC News dataset.
- Validate and clean the dataset before preprocessing.
- Preprocess news article text using NLTK.
- Implement TF-IDF + MLP classification.
- Implement Word2Vec + MLP classification.
- Evaluate both classification approaches.
- Generate training, validation, and confusion-matrix visualisations.
- Measure computational efficiency.
- Display predictions for randomly selected testing samples.
"""

import re
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------------
# Project Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "BBCNews.csv"

TFIDF_OUTPUT_DIR = BASE_DIR / "output" / "tfidf"
WORD2VEC_OUTPUT_DIR = BASE_DIR / "output" / "word2vec"

RANDOM_STATE = 42

WORD2VEC_VECTOR_SIZE = 100


# Convert the original dataset labels to the category names used
# throughout the assignment.
CATEGORY_NAMES = {
    "business": "Business",
    "entertainment": "Entertainment",
    "politics": "Politics",
    "sport": "Sport",
    "tech": "Technology",
}


# ---------------------------------------------------------------------------
# Dataset Loading and Inspection
# ---------------------------------------------------------------------------

def load_data(file_path):
    """
    Load the BBC News dataset from a CSV file.

    Args:
        file_path (Path): Path to the BBC News CSV file.

    Returns:
        pandas.DataFrame: Loaded BBC News dataset.
    """

    data = pd.read_csv(file_path)

    return data


def inspect_data(data):
    """
    Display detailed information about the structure and quality
    of the original BBC News dataset.

    This function is retained for detailed testing and inspection,
    although the final application displays a shorter summary.

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


# ---------------------------------------------------------------------------
# Dataset Cleaning
# ---------------------------------------------------------------------------

def clean_data(data):
    """
    Clean the BBC News dataset before text preprocessing.

    The function:
    - Removes exact duplicate rows.
    - Normalises category labels to the five assignment categories.
    - Verifies that all categories were recognised.

    Args:
        data (pandas.DataFrame): Original BBC News dataset.

    Returns:
        pandas.DataFrame: Cleaned BBC News dataset.
    """

    # Create a copy so the original DataFrame is not modified.
    cleaned_data = data.copy()

    # Remove exact duplicate records before splitting the dataset.
    cleaned_data = (
        cleaned_data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Convert the original category labels to consistent names.
    cleaned_data["category"] = (
        cleaned_data["category"]
        .map(CATEGORY_NAMES)
    )

    # Ensure that every category was successfully recognised.
    if cleaned_data["category"].isnull().any():
        raise ValueError(
            "The dataset contains an unrecognised category label."
        )

    return cleaned_data


def display_cleaning_results(
    original_data,
    cleaned_data,
):
    """
    Display detailed results from dataset cleaning.

    Args:
        original_data (pandas.DataFrame): Dataset before cleaning.
        cleaned_data (pandas.DataFrame): Dataset after cleaning.
    """

    removed_duplicates = (
        len(original_data)
        - len(cleaned_data)
    )

    print("\nDataset Cleaning:")
    print(
        f"Original rows: "
        f"{len(original_data)}"
    )
    print(
        f"Duplicate rows removed: "
        f"{removed_duplicates}"
    )
    print(
        f"Rows after cleaning: "
        f"{len(cleaned_data)}"
    )

    print("\nCleaned category distribution:")
    print(
        cleaned_data["category"]
        .value_counts()
    )


# ---------------------------------------------------------------------------
# NLTK Setup and Text Preprocessing
# ---------------------------------------------------------------------------

def ensure_nltk_resources():
    """
    Ensure the NLTK resources required for preprocessing are available.

    The English stop-word corpus is downloaded automatically if it is
    not already installed.
    """

    resources = {
        "corpora/stopwords": "stopwords",
    }

    for resource_path, download_name in resources.items():

        try:
            nltk.data.find(
                resource_path
            )

        except LookupError:

            print(
                f"Downloading required NLTK resource: "
                f"{download_name}"
            )

            nltk.download(
                download_name,
                quiet=True,
            )


def preprocess_text(
    text,
    stop_words,
):
    """
    Preprocess one BBC News article using NLTK.

    The preprocessing steps include:
    - Converting text to lowercase.
    - Removing punctuation and special characters.
    - Retaining alphabetic and numerical information.
    - Tokenising the text using NLTK.
    - Removing English stop words.

    Numerical information is retained because numbers such as years,
    financial values, percentages, scores, and technology versions may
    contain useful information for news-topic classification.

    Args:
        text (str): Original article text.
        stop_words (set): English stop words to remove.

    Returns:
        list: Preprocessed article tokens.
    """

    # Convert the complete article to lowercase.
    text = text.lower()

    # Remove punctuation and special characters while preserving
    # letters, numbers, and whitespace.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Tokenise the cleaned article using NLTK.
    #
    # preserve_line=True avoids requiring sentence tokenisation because
    # the complete article is already being processed as one sequence.
    tokens = word_tokenize(
        text,
        preserve_line=True,
    )

    # Remove common English stop words.
    tokens = [
        token
        for token in tokens
        if token not in stop_words
    ]

    return tokens


def preprocess_data(data):
    """
    Preprocess every article in the BBC News dataset using NLTK.

    Two new columns are created:
    - tokens: tokenised text used by Word2Vec.
    - processed_text: cleaned text used by TF-IDF.

    Args:
        data (pandas.DataFrame): Cleaned BBC News dataset.

    Returns:
        pandas.DataFrame: Dataset containing preprocessed text.
    """

    ensure_nltk_resources()

    processed_data = data.copy()

    # Load the English stop-word collection only once.
    stop_words = set(
        stopwords.words("english")
    )

    # Apply the same preprocessing pipeline to every article.
    processed_data["tokens"] = (
        processed_data["text"]
        .apply(
            lambda text: preprocess_text(
                text,
                stop_words,
            )
        )
    )

    # Convert the token lists into strings for TF-IDF.
    processed_data["processed_text"] = (
        processed_data["tokens"]
        .apply(" ".join)
    )

    return processed_data


def display_preprocessing_results(data):
    """
    Display detailed preprocessing examples and statistics.

    This function is retained for testing and analysis even though
    the final application displays a shorter startup summary.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.
    """

    print("\nText Preprocessing:")

    print("\nOriginal text example:")
    print(
        data.loc[
            0,
            "text",
        ][:300]
    )

    print("\nProcessed text example:")
    print(
        data.loc[
            0,
            "processed_text",
        ][:300]
    )

    empty_documents = (
        data["processed_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    token_counts = (
        data["tokens"]
        .apply(len)
    )

    print("\nPreprocessing statistics:")

    print(
        f"Empty processed documents: "
        f"{empty_documents}"
    )

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


def display_dataset_summary(
    original_data,
    cleaned_data,
    processed_data,
):
    """
    Display a concise summary of dataset validation and preprocessing.

    Args:
        original_data (pandas.DataFrame): Original BBC News dataset.
        cleaned_data (pandas.DataFrame): Dataset after cleaning.
        processed_data (pandas.DataFrame): Preprocessed dataset.
    """

    duplicate_count = (
        len(original_data)
        - len(cleaned_data)
    )

    missing_values = (
        original_data
        .isnull()
        .sum()
        .sum()
    )

    empty_documents = (
        processed_data["processed_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    category_count = (
        cleaned_data["category"]
        .nunique()
    )

    print("\nBBC News Dataset Preparation")
    print("============================")

    print(
        f"Original articles:        "
        f"{len(original_data)}"
    )

    print(
        f"Duplicate rows removed:   "
        f"{duplicate_count}"
    )

    print(
        f"Cleaned articles:         "
        f"{len(cleaned_data)}"
    )

    print(
        f"News categories:          "
        f"{category_count}"
    )

    print(
        f"Missing values:           "
        f"{missing_values}"
    )

    print(
        f"Empty processed articles: "
        f"{empty_documents}"
    )


# ---------------------------------------------------------------------------
# Dataset Splitting
# ---------------------------------------------------------------------------

def split_dataset(
    data,
    feature_column,
):
    """
    Split a selected feature representation into training, validation,
    and testing subsets.

    The final testing set contains 20% of the complete cleaned dataset.

    The remaining 80% is divided again so that a validation subset can
    be used to monitor model performance during training.

    Stratified splitting preserves the approximate category distribution
    across each subset.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.
        feature_column (str): Column containing the representation to
            split, such as processed_text or tokens.

    Returns:
        tuple: Training, validation, and testing features and labels.
    """

    features = data[
        feature_column
    ]

    labels = data[
        "category"
    ]

    # Keep 20% of the complete cleaned dataset for final testing.
    (
        train_features,
        test_features,
        train_labels,
        test_labels,
    ) = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Split the remaining data again to create the validation subset.
    (
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    ) = train_test_split(
        train_features,
        train_labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_labels,
    )

    return (
        train_features,
        validation_features,
        test_features,
        train_labels,
        validation_labels,
        test_labels,
    )


# ---------------------------------------------------------------------------
# TF-IDF Feature Extraction
# ---------------------------------------------------------------------------

def create_tfidf_features(
    train_texts,
    validation_texts,
    test_texts,
):
    """
    Convert preprocessed BBC News articles into TF-IDF feature vectors.

    The TF-IDF vectorizer is fitted only on training data to prevent
    information from the validation or testing subsets leaking into
    feature generation.

    Args:
        train_texts (pandas.Series): Training article text.
        validation_texts (pandas.Series): Validation article text.
        test_texts (pandas.Series): Testing article text.

    Returns:
        tuple: Vectorizer and transformed feature matrices.
    """

    vectorizer = TfidfVectorizer(
        max_features=5000,
    )

    train_features = (
        vectorizer
        .fit_transform(
            train_texts
        )
    )

    validation_features = (
        vectorizer
        .transform(
            validation_texts
        )
    )

    test_features = (
        vectorizer
        .transform(
            test_texts
        )
    )

    return (
        vectorizer,
        train_features,
        validation_features,
        test_features,
    )


# ---------------------------------------------------------------------------
# TF-IDF + MLP Training
# ---------------------------------------------------------------------------

def train_tfidf_mlp(
    train_features,
    validation_features,
    train_labels,
    validation_labels,
    epochs=30,
):
    """
    Train an MLP classifier using TF-IDF features.

    The classifier is trained incrementally so training loss,
    validation loss, training accuracy, and validation accuracy
    can be recorded for every epoch.

    The model with the highest validation accuracy is retained.

    Args:
        train_features: TF-IDF training feature matrix.
        validation_features: TF-IDF validation feature matrix.
        train_labels (pandas.Series): Training labels.
        validation_labels (pandas.Series): Validation labels.
        epochs (int): Number of training epochs.

    Returns:
        tuple: Best trained model and recorded training history.
    """

    model = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        batch_size=32,
        learning_rate_init=0.001,
        random_state=RANDOM_STATE,
    )

    classes = sorted(
        train_labels.unique()
    )

    loss_history = []
    validation_loss_history = []

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

        training_predictions = (
            model.predict(
                train_features
            )
        )

        validation_predictions = (
            model.predict(
                validation_features
            )
        )

        validation_probabilities = (
            model.predict_proba(
                validation_features
            )
        )

        training_accuracy = accuracy_score(
            train_labels,
            training_predictions,
        )

        validation_accuracy = accuracy_score(
            validation_labels,
            validation_predictions,
        )

        validation_loss = log_loss(
            validation_labels,
            validation_probabilities,
            labels=model.classes_,
        )

        loss_history.append(
            model.loss_
        )

        validation_loss_history.append(
            validation_loss
        )

        training_accuracy_history.append(
            training_accuracy
        )

        validation_accuracy_history.append(
            validation_accuracy
        )

        # Retain the model that produces the highest validation accuracy.
        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            best_model = deepcopy(
                model
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} - "
            f"Training Loss: {model.loss_:.4f} - "
            f"Validation Loss: {validation_loss:.4f} - "
            f"Training Accuracy: {training_accuracy:.4f} - "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    history = {
        "loss": loss_history,
        "validation_loss": validation_loss_history,
        "training_accuracy": training_accuracy_history,
        "validation_accuracy": validation_accuracy_history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
    }

    return (
        best_model,
        history,
    )


def display_tfidf_setup(
    train_labels,
    validation_labels,
    test_labels,
    vectorizer,
):
    """
    Display information about the TF-IDF experiment.
    """

    print(
        "\nTF-IDF + MLP Dataset Split:"
    )

    print(
        f"Training samples: "
        f"{len(train_labels)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_labels)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_labels)}"
    )

    print(
        f"TF-IDF vocabulary size: "
        f"{len(vectorizer.get_feature_names_out())}"
    )


# ---------------------------------------------------------------------------
# TF-IDF Training Figures
# ---------------------------------------------------------------------------

def save_tfidf_training_curves(
    history,
):
    """
    Save TF-IDF + MLP loss and accuracy curves.

    Args:
        history (dict): Training and validation history.
    """

    TFIDF_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history["loss"]) + 1,
    )

    # -----------------------------------------------------------------------
    # Training and Validation Loss
    # -----------------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.title(
        "TF-IDF + MLP Training and Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        TFIDF_OUTPUT_DIR
        / "training_loss.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------------------------
    # Training and Validation Accuracy
    # -----------------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

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

    plt.title(
        "TF-IDF + MLP Training and Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        TFIDF_OUTPUT_DIR
        / "accuracy_curve.png",
        dpi=300,
    )

    plt.close()

    print(
        "\nTF-IDF training curves saved to: "
        f"{TFIDF_OUTPUT_DIR}"
    )


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    test_features,
    test_labels,
    model_name,
):
    """
    Evaluate a trained classifier on the final testing set.

    Args:
        model (MLPClassifier): Trained MLP classifier.
        test_features: Testing feature matrix.
        test_labels (pandas.Series): True testing labels.
        model_name (str): Name displayed in the evaluation output.

    Returns:
        tuple: Predictions and calculated performance metrics.
    """

    predictions = model.predict(
        test_features
    )

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

    print(
        f"\n{model_name} Test Results:"
    )

    print(
        f"Accuracy:  "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall:    "
        f"{recall:.4f}"
    )

    print(
        f"F1-score:  "
        f"{f1_score:.4f}"
    )

    print(
        "\nClassification Report:"
    )

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

    return (
        predictions,
        metrics,
    )


def save_confusion_matrix(
    test_labels,
    predictions,
    output_directory,
    title,
):
    """
    Generate and save a confusion matrix.

    Args:
        test_labels (pandas.Series): True testing labels.
        predictions (numpy.ndarray): Predicted testing labels.
        output_directory (Path): Directory where the figure is saved.
        title (str): Figure title.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = sorted(
        test_labels.unique()
    )

    display = (
        ConfusionMatrixDisplay
        .from_predictions(
            test_labels,
            predictions,
            labels=labels,
            values_format="d",
            xticks_rotation=45,
        )
    )

    display.figure_.set_size_inches(
        9,
        7,
    )

    display.ax_.set_title(
        title
    )

    display.figure_.tight_layout()

    display.figure_.savefig(
        output_directory
        / "confusion_matrix.png",
        dpi=300,
    )

    plt.close(
        display.figure_
    )


# ---------------------------------------------------------------------------
# Random TF-IDF Predictions
# ---------------------------------------------------------------------------

def display_random_tfidf_predictions(
    model,
    vectorizer,
    test_texts,
    test_labels,
    sample_count=10,
):
    """
    Display predictions for randomly selected TF-IDF testing articles.

    Args:
        model (MLPClassifier): Trained classifier.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        test_texts (pandas.Series): Testing article text.
        test_labels (pandas.Series): True testing labels.
        sample_count (int): Number of random samples.
    """

    sample_indices = (
        test_texts
        .sample(
            n=sample_count,
            random_state=RANDOM_STATE,
        )
        .index
    )

    print(
        "\n10 Random TF-IDF + MLP Predictions:"
    )

    for number, index in enumerate(
        sample_indices,
        start=1,
    ):

        article_text = (
            test_texts.loc[
                index
            ]
        )

        actual_category = (
            test_labels.loc[
                index
            ]
        )

        article_features = (
            vectorizer
            .transform(
                [article_text]
            )
        )

        predicted_category = (
            model.predict(
                article_features
            )[0]
        )

        print(
            f"\nSample {number}"
        )

        print(
            f"Actual:    "
            f"{actual_category}"
        )

        print(
            f"Predicted: "
            f"{predicted_category}"
        )

        print(
            "Text:      "
            f"{article_text[:150]}..."
        )


# ---------------------------------------------------------------------------
# TF-IDF Pipeline
# ---------------------------------------------------------------------------

def run_tfidf_mlp(data):
    """
    Run the complete TF-IDF + MLP classification pipeline.

    The function records feature-generation time, MLP training time,
    and total pipeline execution time.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.

    Returns:
        dict: Model results, evaluation metrics, and timing information.
    """

    pipeline_start = perf_counter()

    (
        train_texts,
        validation_texts,
        test_texts,
        train_labels,
        validation_labels,
        test_labels,
    ) = split_dataset(
        data,
        "processed_text",
    )

    # -----------------------------------------------------------------------
    # TF-IDF Feature Generation Timing
    # -----------------------------------------------------------------------

    feature_start = perf_counter()

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

    feature_time = (
        perf_counter()
        - feature_start
    )

    display_tfidf_setup(
        train_labels,
        validation_labels,
        test_labels,
        vectorizer,
    )

    # -----------------------------------------------------------------------
    # MLP Training Timing
    # -----------------------------------------------------------------------

    training_start = perf_counter()

    model, history = train_tfidf_mlp(
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    )

    training_time = (
        perf_counter()
        - training_start
    )

    print(
        "\nBest TF-IDF validation result:"
    )

    print(
        f"Epoch: "
        f"{history['best_epoch']}"
    )

    print(
        f"Validation Accuracy: "
        f"{history['best_validation_accuracy']:.4f}"
    )

    save_tfidf_training_curves(
        history
    )

    predictions, metrics = evaluate_model(
        model,
        test_features,
        test_labels,
        "TF-IDF + MLP",
    )

    save_confusion_matrix(
        test_labels,
        predictions,
        TFIDF_OUTPUT_DIR,
        "TF-IDF + MLP Confusion Matrix",
    )

    display_random_tfidf_predictions(
        model,
        vectorizer,
        test_texts,
        test_labels,
    )

    total_time = (
        perf_counter()
        - pipeline_start
    )

    print(
        "\nTF-IDF + MLP Computational Efficiency:"
    )

    print(
        f"Feature generation time: "
        f"{feature_time:.4f} seconds"
    )

    print(
        f"MLP training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Total pipeline time: "
        f"{total_time:.4f} seconds"
    )

    print(
        "\nTF-IDF + MLP pipeline complete."
    )

    return {
        "model": model,
        "vectorizer": vectorizer,
        "history": history,
        "metrics": metrics,
        "predictions": predictions,
        "feature_time": feature_time,
        "training_time": training_time,
        "total_time": total_time,
    }


# ---------------------------------------------------------------------------
# Word2Vec Feature Generation
# ---------------------------------------------------------------------------

def create_document_vector(
    tokens,
    word2vec_model,
):
    """
    Create one article vector by averaging recognised Word2Vec vectors.

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

    # Return a zero vector if none of the article words exist
    # in the trained Word2Vec vocabulary.
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
    into an averaged document vector.

    Word2Vec is trained only on the training subset to prevent
    information leakage.

    Args:
        train_tokens (pandas.Series): Training token lists.
        validation_tokens (pandas.Series): Validation token lists.
        test_tokens (pandas.Series): Testing token lists.

    Returns:
        tuple: Word2Vec model and document feature matrices.
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

    train_features = np.vstack(
        [
            create_document_vector(
                tokens,
                word2vec_model,
            )
            for tokens in train_tokens
        ]
    )

    validation_features = np.vstack(
        [
            create_document_vector(
                tokens,
                word2vec_model,
            )
            for tokens in validation_tokens
        ]
    )

    test_features = np.vstack(
        [
            create_document_vector(
                tokens,
                word2vec_model,
            )
            for tokens in test_tokens
        ]
    )

    return (
        word2vec_model,
        train_features,
        validation_features,
        test_features,
    )


# ---------------------------------------------------------------------------
# Word2Vec + MLP Training
# ---------------------------------------------------------------------------

def train_word2vec_mlp(
    train_features,
    validation_features,
    train_labels,
    validation_labels,
    epochs=30,
):
    """
    Train an MLP classifier using averaged Word2Vec article vectors.

    The classifier is trained incrementally so training loss,
    validation loss, training accuracy, and validation accuracy
    can be recorded for every epoch.

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

    classes = sorted(
        train_labels.unique()
    )

    loss_history = []
    validation_loss_history = []

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

        training_predictions = (
            model.predict(
                train_features
            )
        )

        validation_predictions = (
            model.predict(
                validation_features
            )
        )

        validation_probabilities = (
            model.predict_proba(
                validation_features
            )
        )

        training_accuracy = accuracy_score(
            train_labels,
            training_predictions,
        )

        validation_accuracy = accuracy_score(
            validation_labels,
            validation_predictions,
        )

        validation_loss = log_loss(
            validation_labels,
            validation_probabilities,
            labels=model.classes_,
        )

        loss_history.append(
            model.loss_
        )

        validation_loss_history.append(
            validation_loss
        )

        training_accuracy_history.append(
            training_accuracy
        )

        validation_accuracy_history.append(
            validation_accuracy
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            best_model = deepcopy(
                model
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} - "
            f"Training Loss: {model.loss_:.4f} - "
            f"Validation Loss: {validation_loss:.4f} - "
            f"Training Accuracy: {training_accuracy:.4f} - "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    history = {
        "loss": loss_history,
        "validation_loss": validation_loss_history,
        "training_accuracy": training_accuracy_history,
        "validation_accuracy": validation_accuracy_history,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
    }

    return (
        best_model,
        history,
    )


# ---------------------------------------------------------------------------
# Word2Vec Training Figures
# ---------------------------------------------------------------------------

def save_word2vec_training_curves(
    history,
):
    """
    Save Word2Vec + MLP loss and accuracy curves.

    Args:
        history (dict): Training and validation history.
    """

    WORD2VEC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(history["loss"]) + 1,
    )

    # -----------------------------------------------------------------------
    # Training and Validation Loss
    # -----------------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["loss"],
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.title(
        "Word2Vec + MLP Training and Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        WORD2VEC_OUTPUT_DIR
        / "training_loss.png",
        dpi=300,
    )

    plt.close()

    # -----------------------------------------------------------------------
    # Training and Validation Accuracy
    # -----------------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

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

    plt.title(
        "Word2Vec + MLP Training and Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        WORD2VEC_OUTPUT_DIR
        / "accuracy_curve.png",
        dpi=300,
    )

    plt.close()

    print(
        "\nWord2Vec training curves saved to: "
        f"{WORD2VEC_OUTPUT_DIR}"
    )


# ---------------------------------------------------------------------------
# Random Word2Vec Predictions
# ---------------------------------------------------------------------------

def display_random_word2vec_predictions(
    model,
    word2vec_model,
    test_tokens,
    test_labels,
    sample_count=10,
):
    """
    Display predictions for randomly selected Word2Vec testing articles.

    Args:
        model (MLPClassifier): Trained classifier.
        word2vec_model (Word2Vec): Trained Word2Vec model.
        test_tokens (pandas.Series): Testing article token lists.
        test_labels (pandas.Series): True testing labels.
        sample_count (int): Number of random samples.
    """

    sample_indices = (
        test_tokens
        .sample(
            n=sample_count,
            random_state=RANDOM_STATE,
        )
        .index
    )

    print(
        "\n10 Random Word2Vec + MLP Predictions:"
    )

    for number, index in enumerate(
        sample_indices,
        start=1,
    ):

        article_tokens = (
            test_tokens.loc[
                index
            ]
        )

        actual_category = (
            test_labels.loc[
                index
            ]
        )

        article_vector = (
            create_document_vector(
                article_tokens,
                word2vec_model,
            )
            .reshape(
                1,
                -1,
            )
        )

        predicted_category = (
            model.predict(
                article_vector
            )[0]
        )

        article_preview = " ".join(
            article_tokens[:25]
        )

        print(
            f"\nSample {number}"
        )

        print(
            f"Actual:    "
            f"{actual_category}"
        )

        print(
            f"Predicted: "
            f"{predicted_category}"
        )

        print(
            f"Text:      "
            f"{article_preview}..."
        )


# ---------------------------------------------------------------------------
# Word2Vec Pipeline
# ---------------------------------------------------------------------------

def run_word2vec_mlp(data):
    """
    Run the complete Word2Vec + MLP classification pipeline.

    The function records feature-generation time, MLP training time,
    and total pipeline execution time.

    Args:
        data (pandas.DataFrame): Preprocessed BBC News dataset.

    Returns:
        dict: Model results, evaluation metrics, and timing information.
    """

    pipeline_start = perf_counter()

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

    # -----------------------------------------------------------------------
    # Word2Vec Feature Generation Timing
    # -----------------------------------------------------------------------

    feature_start = perf_counter()

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

    feature_time = (
        perf_counter()
        - feature_start
    )

    print(
        "\nWord2Vec + MLP Dataset Split:"
    )

    print(
        f"Training samples: "
        f"{len(train_labels)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_labels)}"
    )

    print(
        f"Testing samples: "
        f"{len(test_labels)}"
    )

    print(
        f"Word2Vec vocabulary size: "
        f"{len(word2vec_model.wv)}"
    )

    print(
        f"Word2Vec vector size: "
        f"{word2vec_model.vector_size}"
    )

    # -----------------------------------------------------------------------
    # MLP Training Timing
    # -----------------------------------------------------------------------

    training_start = perf_counter()

    model, history = train_word2vec_mlp(
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    )

    training_time = (
        perf_counter()
        - training_start
    )

    print(
        "\nBest Word2Vec validation result:"
    )

    print(
        f"Epoch: "
        f"{history['best_epoch']}"
    )

    print(
        f"Validation Accuracy: "
        f"{history['best_validation_accuracy']:.4f}"
    )

    save_word2vec_training_curves(
        history
    )

    predictions, metrics = evaluate_model(
        model,
        test_features,
        test_labels,
        "Word2Vec + MLP",
    )

    save_confusion_matrix(
        test_labels,
        predictions,
        WORD2VEC_OUTPUT_DIR,
        "Word2Vec + MLP Confusion Matrix",
    )

    display_random_word2vec_predictions(
        model,
        word2vec_model,
        test_tokens,
        test_labels,
    )

    total_time = (
        perf_counter()
        - pipeline_start
    )

    print(
        "\nWord2Vec + MLP Computational Efficiency:"
    )

    print(
        f"Feature generation time: "
        f"{feature_time:.4f} seconds"
    )

    print(
        f"MLP training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Total pipeline time: "
        f"{total_time:.4f} seconds"
    )

    print(
        "\nWord2Vec + MLP pipeline complete."
    )

    return {
        "model": model,
        "word2vec_model": word2vec_model,
        "history": history,
        "metrics": metrics,
        "predictions": predictions,
        "feature_time": feature_time,
        "training_time": training_time,
        "total_time": total_time,
    }


# ---------------------------------------------------------------------------
# Application Menu
# ---------------------------------------------------------------------------

def display_menu():
    """
    Display the BBC News classification menu.

    Returns:
        str: User-selected menu option.
    """

    print(
        "\nBBC News Text Classification"
    )

    print(
        "============================="
    )

    print(
        "1. TF-IDF + MLP"
    )

    print(
        "2. Word2Vec + MLP"
    )

    print(
        "3. Exit Application"
    )

    return input(
        "\nSelect an option: "
    ).strip()


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    """
    Run the BBC News Text Classification application.

    The program loads and validates the BBC News dataset, removes
    duplicate records, preprocesses article text using NLTK, and
    provides a menu for running either classification approach.
    """

    # -----------------------------------------------------------------------
    # Load the original dataset.
    # -----------------------------------------------------------------------

    data = load_data(
        DATA_FILE
    )

    # -----------------------------------------------------------------------
    # Clean the dataset.
    # -----------------------------------------------------------------------

    cleaned_data = clean_data(
        data
    )

    # -----------------------------------------------------------------------
    # Preprocess the article text.
    # -----------------------------------------------------------------------

    processed_data = preprocess_data(
        cleaned_data
    )

    # Display a concise startup summary rather than the complete
    # development diagnostics.
    display_dataset_summary(
        data,
        cleaned_data,
        processed_data,
    )

    # Detailed diagnostic functions remain available for testing:
    #
    # inspect_data(data)
    #
    # display_cleaning_results(
    #     data,
    #     cleaned_data,
    # )
    #
    # display_preprocessing_results(
    #     processed_data
    # )

    # -----------------------------------------------------------------------
    # Application Menu
    # -----------------------------------------------------------------------

    while True:

        choice = display_menu()

        if choice == "1":

            print(
                "\nRunning TF-IDF + MLP classification...\n"
            )

            run_tfidf_mlp(
                processed_data
            )

        elif choice == "2":

            print(
                "\nRunning Word2Vec + MLP classification...\n"
            )

            run_word2vec_mlp(
                processed_data
            )

        elif choice == "3":

            print(
                "\nExiting BBC News Text Classification."
            )

            break

        else:

            print(
                "\nInvalid option. "
                "Please select 1, 2, or 3."
            )


if __name__ == "__main__":
    main()