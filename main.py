import os
import pickle
from sklearn.externals import joblib
from model.create_model import review_messages

MODEL_FILE = 'model/model.pki'

def predict(mail):
    # Load model
    try:
        with open(MODEL_FILE, 'rb') as f:
            NB_spam_model = open(MODEL_FILE,'rb')
            clf = joblib.load(NB_spam_model)
    except OSError as e:
        print(e)
        from model.create_model import main
        clf = main()
    except Exception as e:
        raise Exception(f'Error: {e}')

    features = review_messages(mail)
    result = clf.classify(features)) 
    return result
