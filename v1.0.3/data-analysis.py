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
                
def contentProcessing(text):
    # Drop symbols
    job_text = re.sub("[^a-zA-Z.+]", "  ", text)

    # Lowercase and split into words
    job_tokens = job_text.lower().split()

    # Filter out stop words
    stop_words = stopwords.words("english")

    stop_words.append("k")
    stop_words.append("new")
    stop_words.append("you")
    stop_words.append("we")
    stop_words.append("job")
    stop_words.append("u")
    stop_words.append("it'")
    stop_words.append("'s")
    stop_words.append("n't")
    stop_words.append("mr.")
    stop_words.append(".")
    stop_words.append("+")

    stop_words = set(stop_words)

    job_tokens = [word for word in job_tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    job_tokens = [lemmatizer.lemmatize(word) for word in job_tokens]

    # Stem
    #stemmer = PorterStemmer()

    #job_tokens = [stemmer.stem(word) for word in job_tokens]
    
    tokens = [word.encode('utf-8', 'strict') for word in job_tokens]

    return tokens

# Get word frequency distribution
def freqDist(tokens):
    return FreqDist(tokens).most_common()

def keywords(category):
    tokens = data[data['category'] == category]['tokens']
    allTokens = []

    for token_list in tokens:
        allTokens += token_list

    counter = Counter(allTokens)
    return counter.most_common(10)

if __name__ == '__main__':
    # Load data from file to dataframe
    data = pd.read_csv('./jobs.csv')

    # Check size of document (r, c)
    print 'data shape: ', data.shape

    # Create graph to show how many jobs/category
    #data.category.value_counts().plot(kind='bar', grid=True, figsize=(3, 5))
    #pylab.show()

    tokens = data['descr'].map(contentProcessing)

    data['tokens'] = tokens

    for descr, tokens in zip(data['descr'].head(5), data['tokens'].head(5)):
        print('description:', descr)
        print('tokens:', tokens)
        print()

    for category in set(data['category']):
        print('category:', category)
        print('top 10 keywords:', keywords(category))
        print('---')