
# Assignment question 2
The aim of this question is to classify the dataset using Naive Bayes Classifier.





## Installation

Install the nltk packages before preprocessing

```bash
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

```python
import numpy as np 
import pandas as pd
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stopword1=stopwords.words('english')
import requests
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# returns 'preprocessed dataset'
# returns 'X_train,Y_train,X_test,Y_test'
# returns 'Naive Bayes Classifier'
# returns 'accuracy,precision,recall,F1-score'

    