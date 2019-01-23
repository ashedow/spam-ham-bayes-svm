import os
import random
import pickle
from pandas import DataFrame
from numpy import zeros
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer, classify, NaiveBayesClassifier, stem
# from nltk.classify.scikitlearn import SklearnClassifier
from string import punctuation
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

SPAM_PATH = 'data/spam/'
HAM_PATH = 'data/ham/'
TEST_SPAM_PATH = 'data/test_spam'
TEST_HAM_PATH = 'data/test_ham'

def create_list(path):
    file_list = os.listdir(path)
    lst = []
    for i in file_list:
        f = open(path + i, 'r', encoding="utf8", errors='ignore')
        lst.append(f.read())
    f.close()
    return lst

def process_email(texts, custom_stopwords=[]):
    emails = []
    lemmatizer = WordNetLemmatizer()
    my_stopwords = stopwords.words('english') + list(punctuation) + custom_stopwords
    for sentence in texts:
        tokens = word_tokenize(sentence)
        processed_email = []
        for word in tokens:
            lemma = lemmatizer.lemmatize(word.lower())
            if lemma not in my_stopwords and len(lemma)>2:
                processed_email.append(lemma)
        emails = emails + processed_email
    return emails

def review_messages(texts):
    stoplist = stopwords.words('english')
    stoplist.append('Subject')
    stemmer = stem.SnowballStemmer('english')

    words = process_email(texts)
    msg = []
    for word in words:
        if word not in stoplist:
            msg.append(word)
    # using a stemmer
    msg = " ".join([stemmer.stem(word) for word in msg])
    return msg

def data_frame(text_mail, classification):
    rows = []
    index = []
    for text, message in text_mail:
        rows.append({'message': message, 'class': classification})
        index.append(text)

    return DataFrame(rows, index=index)

def prepare_train():
    data = DataFrame({'message': [], 'class': []})
    # Loading and clean the email-dataset 
    spam_data = create_list(SPAM_PATH)
    ham_data = create_list(HAM_PATH)
    spam = process_email(spam_data, ['subject'])
    ham = process_email(ham_data, ['subject'])
    spam = review_messages(spam)
    ham = review_messages(ham)

    data = data.append(data_frame(spam, 'spam'))
    data = data.append(data_frame(ham, 'ham'))
    random.shuffle(data)    
    return data

def prepare_test():
    test_data = DataFrame({'message': [], 'class': []})
    test_spam = review_messages(create_list(TEST_SPAM_PATH))
    test_ham = review_messages(create_list(TEST_HAM_PATH))

    frame_spam = data_frame(test_ham, 'ham')
    frame_ham = data_frame(test_spam, 'ham')

    test_data = test_data.append(frame_ham)
    test_data = test_data.append(frame_spam)
    return test_data

def testing_models(data, test_data):

    vectorizer = CountVectorizer(min_df=1)
    counts = vectorizer.fit_transform(data['message'].values)
    targets = data['class'].values
    param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
    
    # Training SVM and Naive bayes classifier

    NB_model = MultinomialNB()
    GNB_model = GaussianNB()
    BNB_model = BernoulliNB()
    LinearSVM_model = svm.LinearSVC()
    SVM_model = svm.SVC()
    grid = GridSearchCV(svm.SVC(), param_grid, verbose=3)

    models = {'NB_model': NB_model, 'GNB_model': GNB_model, 'BNB_model': BNB_model,
                'LinearSVM_model': LinearSVM_model, 'SVM_model': SVM_model, 'grid': grid}
    
    models = {model.fit(counts, targets) for name, model in models.items()}

    #check accurasy
    model_predict = {model.predict(test_data) for name, model in models.items()}
    model_acc = {model.accuracy_score(test_data) for name, model in models.items()}


    max_value = max(model_predict.values())
    max_accuracy = {k: v for k, v in model_predict.items() if v == max_value}
    
    print('MODEL_PREDICT', model_predict)
    print('MODEL_ACC',  model_acc)

    return {k: models[k] for k,v in max_accuracy.keys()}

def pickle_model(clf):
    filename = 'model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)

def main():
    data = prepare_train()
    test_data = prepare_test()

    # Return the model with best accuracy
    clf = testing_models(data, test_data)

    print("SHOW_MOST_INF_FEATURES", clf.show_most_informative_features(20))
    pickle_model(clf)
    return clf