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

The dataset was downloaded from Kaggle as permitted by the assignment
specification and stored locally as:

`data/BBCNews.csv`

## Data Cleaning

Before preprocessing, the application:

- Checks for missing values
- Checks for duplicate records
- Removes exact duplicate articles
- Normalises the category names used by the dataset

The original dataset contains 2,225 rows.

After removing 99 exact duplicate records, 2,126 unique articles remain.

The cleaned category distribution is:

| Category | Articles |
|---|---:|
| Sport | 504 |
| Business | 503 |
| Politics | 403 |
| Entertainment | 369 |
| Technology | 347 |

## Text Preprocessing

The application performs the preprocessing required for Problem 2:

- Converts article text to lowercase
- Removes punctuation and special characters
- Tokenises text into individual words
- Removes common English stop words

Two representations are retained after preprocessing:

- `tokens` - Token lists that will be used by Word2Vec
- `processed_text` - Cleaned text strings that will be used by TF-IDF

Preprocessing validation confirmed that no articles became empty after
cleaning.

The current preprocessing statistics are:

- Average tokens per article: `213.17`
- Minimum tokens in an article: `45`
- Maximum tokens in an article: `2060`

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

Word2Vec is trained only on the training articles to prevent information
from the validation and testing sets leaking into feature generation.

Each article is represented by averaging the Word2Vec vectors of the
recognised words contained in the article.

The trained Word2Vec vocabulary contains 15,146 words.

### MLP Configuration

The Word2Vec feature vectors are classified using the same MLP architecture
used for the TF-IDF experiment:

- Hidden layer: 128 neurons
- Activation: ReLU
- Optimiser: Adam
- Batch size: 32
- Initial learning rate: 0.001
- Maximum training epochs: 30
- Random seed: 42

The model with the highest validation accuracy is retained for final
testing.

### Word2Vec Results

The best validation result occurred at epoch 7:

- Validation accuracy: 96.76%

Final testing performance:

| Metric | Result |
|---|---:|
| Accuracy | 96.01% |
| Precision | 96.24% |
| Recall | 96.01% |
| F1-score | 96.02% |

The model continued improving its training accuracy after epoch 7 while
validation performance stopped improving, indicating mild overfitting.

### Generated Word2Vec Figures

The application generates:

`output/word2vec/training_loss.png`

and:

`output/word2vec/accuracy_curve.png`
