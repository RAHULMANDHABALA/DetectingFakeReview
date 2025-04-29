# Fake Product Review Detection System

## Overview
In the modernization process, humans have always purchased desirable products and commodities. We frequently give advice on things to buy and to avoid to our friends, family, and acquaintances based on our own experiences. Similar to this, when we want to purchase something we have never done before, we speak with others who have some knowledge in that field.

Manufacturers have also relied on this technique of getting client reviews to choose the products or product features that will best delight consumers.

But in the age of digital technology, it has evolved into online reviews. It has become vital to concentrate on such online components of the business as a result of the development of E-Commerce in these modern times. Thankfully, all online retailers have started using review systems for their items. With so many individuals linked online and living in various locations around the world, it is becoming a challenge to maintain their reviews and organizing them.

## Algorithms Used

### Support Vector Classifier (SVC)
An SVM is a type of Machine Learning Algorithm that is used mostly for classification methods. An SVM works in a way that it produces a hyperplane that divides two classes. In high-dimensional space, it can produce a hyperplane or collection of hyperplanes. This hyperplane can also be utilized for regression or classification.

**Algorithm:**
1. Choose the hyperplane that best divides the class
2. Determine the Margin (distance between the planes and the data)
3. Choose the class with the largest margin

### Logistic Regression
Logistic regression is a supervised learning algorithm that estimates the probability of the dependent variable based on the independent variable.

**Key Points:**
- Classifies data in binary form (0 for negative, 1 for positive)
- Uses a sigmoid function to predict probabilities
- Based on linear regression model

### Random Forest Classifier
An ensemble learning technique that can be used for classification and regression tasks.

**Advantages:**
- Higher accuracy compared to other models
- Can handle large datasets
- Reduces variance of Decision Trees

**Algorithm:**
1. Select "R" features from total features "m" where R>M
2. Node employs the most effective split point
3. Choose optimal split to divide node into sub-nodes
4. Repeat until reaching "I" number of nodes
5. Build forest by adding "a" number of trees

## Project Prerequisites
- Python 3.6+
- Jupyter Notebook (recommended)
- Required modules:
  ```bash
  pip install numpy pandas seaborn scikit-learn nltk

## Dataset
The dataset contains product reviews with labels indicating whether they are genuine (OR) or computer-generated (CG).

## Implementation Steps

### 1. Import Required Libraries
```python
import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
```

### 2. Load and Prepare Dataset
```python
dataframe = pd.read_csv('dataset.csv')
dataframe.drop('Unnamed: 0', axis=1, inplace=True)
dataframe.dropna(inplace=True)
dataframe['length'] = dataframe['text_'].apply(len)
```

### 3. Text Preprocessing Function
```python
def convertmyTxt(rv):
    np = [c for c in rv if c not in string.punctuation]
    np = ''.join(np)
    return [w for w in np.split() if w.lower() not in stopwords.words('english')]
```

### 4. Train-Test Split
```python
x_train, x_test, y_train, y_test = train_test_split(dataframe['text_'], dataframe['label'], test_size=0.25)
```

### 5. Model Pipelines

#### Random Forest Classifier
```python
pip = Pipeline([
    ('bow', CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier())
])
pip.fit(x_train, y_train)
randomForestClassifier = pip.predict(x_test)
print('Accuracy:', str(np.round(accuracy_score(y_test, randomForestClassifier)*100,2) + '%')
```

#### Support Vector Classifier
```python
pip = Pipeline([
    ('bow', CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC())
])
pip.fit(x_train, y_train)
supportVectorClassifier = pip.predict(x_test)
print('Accuracy:', str(np.round(accuracy_score(y_test, supportVectorClassifier)*100,2) + '%')
```

#### Logistic Regression
```python
pip = Pipeline([
    ('bow', CountVectorizer(analyzer=convertmyTxt)),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])
pip.fit(x_train, y_train)
logisticRegression = pip.predict(x_test)
print('Accuracy:', str(np.round(accuracy_score(y_test, logisticRegression)*100,2) + '%')
```

## Results
- Random Forest Classifier Accuracy: ~83.56%
- Support Vector Classifier Accuracy: ~88.11%
- Logistic Regression Accuracy: ~86.05%

## Summary
This project implements a Fake Product Review Detection System using multiple machine learning models. The system analyzes product reviews to classify them as genuine or computer-generated, helping to maintain the integrity of online review systems.

The Support Vector Classifier showed the best performance among the three models tested, achieving 88.11% accuracy.
```
