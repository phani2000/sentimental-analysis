import nltk
import re
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('words')
nltk.download('stopwords')

#dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset = pd.read_csv("TwitterDataset.csv",encoding='latin1',names=['Liked','QueryId','IssueDate','QueryType','UserId','Review'])
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True,lowercase=True,strip_accents="ascii",stop_words=stopset)

corpus = []

for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

df = pd.DataFrame({'Tweets':corpus})
#df.to_csv('NLP_Restaurant.csv')
df.to_csv('NLP_Twitter.csv')
