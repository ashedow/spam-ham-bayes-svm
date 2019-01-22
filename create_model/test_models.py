import os
import random
import pickle
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer, NaiveBayesClassifier, classify

SPAM_PATH = 'spam/'
HAM_PATH = 'ham/'
TRAIN_PATH = 'test/'

# Raize part of data for select fit model

