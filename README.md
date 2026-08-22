# COIT29225 Assignment 1 - BBC News Classification

Problem 2 of COIT29225 Neural Networks & Deep Learning Assignment 1.

This project implements a Python application for classifying BBC News
articles into five categories using two neural-network classification
approaches:

1. TF-IDF + Multilayer Perceptron (MLP)
2. Word2Vec + Multilayer Perceptron (MLP)

The application includes dataset validation, NLTK text preprocessing,
reproducible dataset splitting, model training, validation-based model
selection, final testing, performance metrics, training visualisations,
confusion matrices, random test predictions, and computational-efficiency
measurements.

## Dataset

The project uses the BBC News text classification dataset containing
2,225 news articles across five categories:

- Business
- Entertainment
- Politics
- Sport
- Technology

The dataset contains two columns:

- `category` - The news article category
- `text` - The news article text

The dataset is stored locally as:

`data/BBCNews.csv`

## Data Cleaning

Before text preprocessing, the application:

- Checks the dataset structure
- Checks for missing values
- Checks for duplicate records
- Removes exact duplicate articles
- Normalises the category labels
- Verifies that all category labels are recognised

The original dataset contains 2,225 articles.

After removing 99 exact duplicate records, 2,126 unique articles remain.

The cleaned category distribution is:

| Category | Articles |
|---|---:|
| Sport | 504 |
| Business | 503 |
| Politics | 403 |
| Entertainment | 369 |
| Technology | 347 |

The final dataset validation confirms:

- Missing values: 0
- Empty processed articles: 0
- Number of categories: 5

## Text Preprocessing

The application uses NLTK for text tokenisation and English stop-word
removal.

The final preprocessing pipeline performs the following steps:

1. Convert article text to lowercase
2. Remove punctuation and special characters
3. Retain alphabetic and numerical information
4. Tokenise the text using NLTK `word_tokenize()`
5. Remove English stop words using the NLTK stop-word corpus

Numerical information is retained because values such as years,
financial figures, percentages, scores, and technology-related numbers
may contain useful information for news-topic classification.

Two representations are retained after preprocessing:

- `tokens` - Token lists used by Word2Vec
- `processed_text` - Cleaned text strings used by TF-IDF

The final preprocessing statistics are:

| Statistic | Result |
|---|---:|
| Empty processed articles | 0 |
| Average tokens per article | 226.77 |
| Minimum tokens in an article | 49 |
| Maximum tokens in an article | 2,220 |

## Dataset Split

The cleaned dataset contains 2,126 articles.

Both classification approaches use the same reproducible stratified split:

| Subset | Articles |
|---|---:|
| Training | 1,360 |
| Validation | 340 |
| Testing | 426 |

A fixed random seed of `42` is used for reproducibility.

Stratification preserves the approximate category distribution across the
training, validation, and testing subsets.

The testing data is kept separate from training and model selection.

## TF-IDF + MLP Classification

The first classification approach represents each article using TF-IDF
features before classification with an MLP neural network.

### TF-IDF Configuration

The TF-IDF vectorizer uses:

- Maximum features: 5,000
- Vocabulary size in the validated run: 5,000

The vectorizer is fitted only on the training articles.

Validation and testing articles are transformed using the vocabulary learned
from the training set to prevent information leakage.

### TF-IDF MLP Configuration

The TF-IDF feature vectors are classified using:

- Hidden layer: 128 neurons
- Activation: ReLU
- Optimiser: Adam
- Batch size: 32
- Initial learning rate: 0.001
- Maximum training epochs: 30
- Random seed: 42

Training is performed incrementally so the application can record:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

The model with the highest validation accuracy is retained for final testing.

### TF-IDF Training Behaviour

The best validation result occurred at epoch 11:

- Best validation accuracy: 97.35%

Training accuracy reached 100% early in the training process.

Training loss continued decreasing throughout the 30 epochs, while
validation loss eventually levelled out and began increasing slightly.

This indicates mild overfitting after the model had converged.

### TF-IDF Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 97.18% |
| Precision | 97.27% |
| Recall | 97.18% |
| F1-score | 97.19% |

The TF-IDF model correctly classified 414 of the 426 testing articles.

### TF-IDF Class-Level Observations

The final confusion matrix shows:

- Sport: 101 of 101 articles classified correctly
- Business: 99 of 101 articles classified correctly
- Entertainment: 72 of 74 articles classified correctly
- Politics: 77 of 81 articles classified correctly
- Technology: 65 of 69 articles classified correctly

Sport was the strongest-performing category in the final TF-IDF model.

## Word2Vec + MLP Classification

The second classification approach uses Word2Vec embeddings to represent
the preprocessed BBC News articles before classification with an MLP.

### Word2Vec Configuration

The Word2Vec model uses:

- Vector size: 100
- Context window: 5
- Minimum word count: 2
- Word2Vec training epochs: 20
- Random seed: 42
- Workers: 1 for reproducibility

Word2Vec is trained only on the training articles to prevent information
from the validation and testing sets leaking into feature generation.

Each article is represented by averaging the Word2Vec vectors of the
recognised words contained in the article.

The trained Word2Vec vocabulary in the final validated run contains:

`15,286 words`

### Word2Vec MLP Configuration

The Word2Vec model uses the same MLP architecture as the TF-IDF experiment:

- Hidden layer: 128 neurons
- Activation: ReLU
- Optimiser: Adam
- Batch size: 32
- Initial learning rate: 0.001
- Maximum training epochs: 30
- Random seed: 42

Using the same MLP architecture allows the feature representations to be
compared more fairly.

### Word2Vec Training Behaviour

The best validation result occurred at epoch 7:

- Best validation accuracy: 96.47%

Training accuracy continued increasing after epoch 7 while validation
accuracy stopped improving.

Validation loss later reached its minimum and then began increasing
slightly while training loss continued to decrease.

This indicates mild overfitting after the model had converged.

### Word2Vec Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 96.01% |
| Precision | 96.26% |
| Recall | 96.01% |
| F1-score | 96.01% |

The Word2Vec model correctly classified 409 of the 426 testing articles.

### Word2Vec Class-Level Observations

The final confusion matrix shows:

- Sport: 101 of 101 articles classified correctly
- Business: 99 of 101 articles classified correctly
- Entertainment: 71 of 74 articles classified correctly
- Politics: 78 of 81 articles classified correctly
- Technology: 60 of 69 articles classified correctly

Technology was the most challenging category for the Word2Vec model.

Seven Technology articles were classified as Business and two were
classified as Politics.

## Application Menu

The final application provides the required menu:

```text
BBC News Text Classification
=============================
1. TF-IDF + MLP
2. Word2Vec + MLP
3. Exit Application
```

The dataset is loaded, cleaned, and preprocessed once when the program
starts.

A concise dataset-preparation summary is displayed before the menu.

Invalid selections are rejected and the menu is displayed again.

## Generated Visualisations

The application generates three figures for each classification approach.

```text
output/
├── tfidf/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── training_loss.png
└── word2vec/
    ├── accuracy_curve.png
    ├── confusion_matrix.png
    └── training_loss.png
```

### Accuracy Curves

The accuracy figures show:

- Training accuracy by epoch
- Validation accuracy by epoch

### Loss Curves

The loss figures show:

- Training loss by epoch
- Validation loss by epoch

These figures provide evidence of model convergence and mild overfitting.

### Confusion Matrices

The confusion matrices show the final testing predictions for each news
category and make it possible to identify which categories are most
frequently confused.

## Random Testing Predictions

Each classification approach displays predictions for 10 randomly selected
testing articles.

A fixed random seed is used so the same testing samples can be reproduced.

For each sample, the application displays:

- Actual category
- Predicted category
- A shortened preview of the processed article

The 10 displayed samples are illustrative examples.

Final performance is evaluated using the complete 426-article testing set.

## Computational Efficiency

The application measures:

- Feature-generation time
- MLP training time
- Total pipeline time

The following timings were recorded during the final validated run on the
development computer:

| Measure | TF-IDF + MLP | Word2Vec + MLP |
|---|---:|---:|
| Feature dimensions | 5,000 | 100 |
| Feature generation time | 0.2238 s | 7.8293 s |
| MLP training time | 20.8182 s | 0.5494 s |
| Total pipeline time | 22.2171 s | 9.3070 s |

These timing measurements are machine-dependent and should not be interpreted
as universal benchmark values.

TF-IDF generated its feature representation much faster than Word2Vec.

However, the TF-IDF MLP receives a 5,000-dimensional input representation,
while the Word2Vec MLP receives a compact 100-dimensional dense vector.
Consequently, MLP training was substantially faster for Word2Vec in the
validated run.

Overall, Word2Vec completed the full pipeline faster on the development
computer, while TF-IDF achieved higher classification accuracy.

## Final Model Comparison

The final validated results are:

| Metric | TF-IDF + MLP | Word2Vec + MLP |
|---|---:|---:|
| Best Validation Accuracy | **97.35%** | 96.47% |
| Best Epoch | 11 | 7 |
| Test Accuracy | **97.18%** | 96.01% |
| Precision | **97.27%** | 96.26% |
| Recall | **97.18%** | 96.01% |
| F1-score | **97.19%** | 96.01% |
| Correct Test Predictions | **414 / 426** | 409 / 426 |
| Feature Generation Time | **0.2238 s** | 7.8293 s |
| MLP Training Time | 20.8182 s | **0.5494 s** |
| Total Pipeline Time | 22.2171 s | **9.3070 s** |

TF-IDF + MLP achieved the highest overall classification performance.

The difference in final test accuracy is:

```text
97.18% - 96.01% = 1.17 percentage points
```

TF-IDF appears particularly effective for this topic-classification task
because the BBC News categories contain distinctive vocabulary that can be
strongly represented using term importance.

Word2Vec provides a much smaller dense representation and completed MLP
training considerably faster, but averaging word embeddings can reduce the
influence of highly category-specific terms.

Both approaches nevertheless achieved strong performance.

## Project Structure

```text
coit29225-a1-bbc-news-classification/
├── data/
│   └── BBCNews.csv
├── output/
│   ├── tfidf/
│   │   ├── accuracy_curve.png
│   │   ├── confusion_matrix.png
│   │   └── training_loss.png
│   └── word2vec/
│       ├── accuracy_curve.png
│       ├── confusion_matrix.png
│       └── training_loss.png
├── q2.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dependencies

The project uses:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- Gensim
- NLTK

Install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Running the Application

Run:

```bash
python q2.py
```

The application prepares the dataset and displays the classification menu.

## Final Status

The Problem 2 implementation includes:

- Dataset loading and validation
- Duplicate removal
- Category normalisation
- NLTK tokenisation
- Lowercase conversion
- Punctuation and special-character removal
- Numerical information retention
- NLTK English stop-word removal
- Reproducible stratified dataset splitting
- TF-IDF feature extraction
- TF-IDF + MLP classification
- Word2Vec feature generation
- Averaged Word2Vec document vectors
- Word2Vec + MLP classification
- Validation-based model selection
- Training and validation loss monitoring
- Training and validation accuracy monitoring
- Final test evaluation
- Accuracy, precision, recall, and F1-score metrics
- Classification reports
- 10 random test predictions for each model
- Confusion matrices
- Computational-efficiency measurement
- Menu-driven model selection
- Invalid-input handling
- Exit functionality