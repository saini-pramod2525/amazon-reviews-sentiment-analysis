# Amazon Reviews Sentiment Analysis

## Project Overview

This project applies Natural Language Processing (NLP) and machine learning techniques to classify Amazon product reviews into **Positive, Neutral, and Negative** sentiments.

The goal is to understand customer opinions, address class imbalance, compare different classification approaches, and identify common themes in product reviews.

## Objectives

- Explore and clean Amazon product review data.
- Analyze sentiment distribution and class imbalance.
- Convert review text into numerical features using TF-IDF.
- Train and compare multiple machine learning models.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Experiment with sentiment feature engineering and LSTM-based classification.
- Explore review themes using LDA and NMF topic modelling.
- Generate sentiment predictions for an unlabeled test dataset.

## Dataset

The provided training dataset contains 4,000 review records before cleaning. After removing 58 duplicate rows, 3,942 records remained.

The dataset includes product information, review titles, review text, and sentiment labels.

The sentiment classes are imbalanced, with Positive reviews representing approximately 93.7% of the original training data.

*Note: The original dataset files are not included in this public repository. Please refer to the course materials for data access.*

## Tools and Technologies

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TF-IDF vectorization
- imbalanced-learn (SMOTE)
- VADER sentiment analysis
- TensorFlow/Keras (LSTM experiments)
- LDA and NMF topic modelling

## Models and Results

The following results were obtained during validation experiments:

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Naive Bayes + XGBoost Ensemble | 94.17% | **0.6727** |
| SMOTE + Multinomial Naive Bayes | 91.13% | 0.6300 |
| Class-weighted Linear SVM | **95.06%** | 0.5937 |
| Logistic Regression + VADER Features | 92.52% | 0.6036 |
| XGBoost | 91.76% | 0.5852 |
| VADER | 93.66% | 0.5486 |
| Original LSTM | 93.03% | 0.4613 |
| Tuned LSTM | 87.83% | 0.4136 |

The Naive Bayes and XGBoost ensemble achieved the highest macro F1-score among these experiments, while the class-weighted Linear SVM achieved the highest validation accuracy.

Macro F1-score is particularly useful for this project because it gives equal weight to each sentiment class despite the imbalance in the dataset.

## Additional Evaluation

The saved SMOTE + Multinomial Naive Bayes model was evaluated on the provided labelled test dataset:

- Accuracy: **89.80%**
- Macro F1-score: **0.5497**

Performance varied across sentiment classes, with Positive reviews classified more reliably than Neutral and Negative reviews. This highlights the difficulty of identifying less frequent sentiment categories.

## Topic Modelling

LDA and NMF were used to explore recurring themes in reviews, including:

- Echo and Alexa devices and smart-home functionality
- Tablets, children, and applications
- Product quality, value, and setup
- Kindle devices and reading experiences

## Project Files

- `Amazon_reviews.ipynb` — data preparation, exploratory analysis, modelling, evaluation, and topic modelling.
- `Amazon_Reviews_Sentiment_Analysis_Project_Report1.docx` — project summary report.

## Key Learnings

- Why accuracy alone can be misleading for imbalanced classification problems.
- How TF-IDF represents review text for machine learning.
- How resampling and class weighting can help address class imbalance.
- How to compare traditional machine learning, ensemble models, and LSTM approaches.
- How topic modelling can reveal recurring themes in customer feedback.

## Conclusion

This project demonstrates an end-to-end sentiment analysis workflow, from data cleaning and text preprocessing to model comparison, evaluation, and topic discovery. The experiments highlight the importance of evaluating minority-class performance alongside overall accuracy when analysing customer reviews.
