import pandas as pd
import numpy as np
import nltk
import re
import pickle
import datetime

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score 

#global X, y, corpus, user_inp, review, dataset, stopset, vectorizer, ps

nltk.download('words')
nltk.download('stopwords')

words = set(nltk.corpus.words.words())
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True,lowercase=True,strip_accents="ascii",stop_words=stopset)
ps = PorterStemmer()


a = datetime.datetime.now()


df = pd.read_csv('Dataset/TwitterDatasetNew.csv')

b = datetime.datetime.now()


print(f"Time taken to load .csv file",b-a)



y = df.Liked

X = vectorizer.fit_transform(df['Review'])


print(y.shape,"\n",X.shape)


'''

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=35, random_state=50)

classifier = naive_bayes.MultinomialNB()
classifier.fit(X_train,y_train)



print(f"Accuracy : {roc_auc_score(y_test,classifier.predict_proba(X_test)[:,1])}")
'''

filename = "Dataset/finalized_model.sav"

#pickle.dump(classifier,open(filename,'wb'))


def output(user_inp):
    
    #user_inp=input("Enter your feedback : ")
    user_inp = re.sub('[^a-zA-Z]', ' ',user_inp).lower().split()
    
    user_inp = ' '.join(user_inp)
    
    user_inp = " ".join(w for w in nltk.wordpunct_tokenize(user_inp) \

         if w.lower() in words or not w.isalpha())
    
    user_inp = user_inp.split()
    
    #user_inp = [ps.stem(word) for word in user_inp if not word in set(stopwords.words('english'))]

    user_inp = [word for word in user_inp if not word in set(stopwords.words('english'))]
    
    user_inp = ' '.join(user_inp)
    
    if user_inp != '':

        a = datetime.datetime.now()
        
        loaded_model = pickle.load(open(filename, 'rb'))

        b = datetime.datetime.now()

        print(f"Time taken to load .sav file",b-a)
        
        print("\n",user_inp)

        Twitter_review_array = np.array([user_inp])

        Twitter_review_vector = vectorizer.transform(Twitter_review_array)

        print(loaded_model.predict(Twitter_review_vector))
    
        out = loaded_model.predict(Twitter_review_vector)

        #print(classifier.predict(Twitter_review_vector))
    
        #out = classifier.predict(Twitter_review_vector)

        if out==[0]:
            outin = 'Negative Response'
        else:
            outin = 'Positive Response'
        
        return outin

    else:
        return 'INVALID RESPONSE'

