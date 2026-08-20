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
