import matplotlib
import pylab
import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.probability import FreqDist

from string import punctuation
from collections import Counter

def tokenizer(text):
    # Drop symbols
    text = re.sub("[^a-zA-Z.+]", "  ", text)

    # Lowercase and split into words
    tokens = text.lower().split()

    # Filter out stop words
    stop = stopwords.words("english")

    stop.append("k")
    stop.append("new")
    stop.append("you")
    stop.append("we")
    stop.append("job")
    stop.append("u")
    stop.append("it'")
    stop.append("'s")
    stop.append("n't")
    stop.append("mr.")
    stop.append(".")
    stop.append("+")

    stop = set(stop)
    
    tokens = [word for word in tokens if word not in stop]
    
    # Filter out punctuation
    tokens = [word for word in tokens if word not in punctuation]

    return tokens

def groupTokens(category):
    tokens    = data[data['category'] == category]['tokens']
    all_tokens = []

    for token in tokens:
        all_tokens += token

    return FreqDist(all_tokens).most_common(10)

def shtuff():
    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    job_tokens = [lemmatizer.lemmatize(word) for word in job_tokens]

    # Stem
    #stemmer = PorterStemmer()

    #job_tokens = [stemmer.stem(word) for word in job_tokens]
    
    encoded = [word.encode('utf-8', 'strict') for word in job_tokens]

    return encoded

# Get word frequency distribution
def freqDist(tokens):
    return FreqDist(tokens).most_common()

if __name__ == '__main__':
    # Load data from file to dataframe
    data = pd.read_csv('./jobs.csv')

    # Check size of document (r, c)
    print 'data shape: ', data.shape

    # Create graph to show how many jobs/category
    data.category.value_counts().plot(kind='bar', grid=True, figsize=(3, 5))
    # To show graph, uncomment next line
    # pylab.show()

    # Tokenize text 
    data['tokens'] = data['descr'].map(tokenizer)

    for descr, tokens in zip(data['descr'].head(1), data['tokens'].head(1)):
        print 'description: ', descr, '\n'
        print 'tokens: ', tokens, '\n'

    # Group tokens by category
    for category in set(data['category']):
        print 'category: ', category, '\n'
        print 'top 10 keywords: ', groupTokens(category), '\n'
        print '--------------------------------------'
