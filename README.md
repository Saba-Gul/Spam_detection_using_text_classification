# Spam Detection Using NLP Techniques

This project implements text classification techniques to detect spam messages using Natural Language Processing (NLP) methods. It includes preprocessing steps, model training, evaluation, and performance analysis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to build a machine learning model that can classify text messages as either spam or not spam (ham). It leverages various NLP techniques such as tokenization, stopword removal, stemming, and n-grams vectorization to preprocess the text data. The model performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Dataset

The dataset used in this project consists of text messages labeled as spam or ham. It is preprocessed to remove noise and irrelevant information before model training.

## Preprocessing

Text preprocessing steps include:

- Lowercasing
- Punctuation removal
- Stopword removal
- Tokenization
- Stemming
- N-grams vectorization (unigrams, bigrams, trigrams)

## Models Used

Two classification models are implemented:

1. Logistic Regression
2. Naive Bayes (Multinomial)

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Results

| Algorithm            | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 97.31%   | 100%      | 79.45% | 88.55%   |
| Naive Bayes (Multinomial) | 94.98%   | 74.73%    | 93.15% | 82.93%   |

## Usage

- Modify `Spam_detection_using_text_classification.ipynb` to experiment with different preprocessing techniques or models.
- Use the provided functions and classes to integrate with other applications or pipelines.

## Contributing

Contributions are welcome! Fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

