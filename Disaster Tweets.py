import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import xgboost as xgb


train_data = pd.read_csv('train.csv')
train_keywords = train_data['keyword']
train_data = train_data.drop('keyword', axis = 1)
test_data = pd.read_csv('test.csv')
test_data = test_data.drop('keyword', axis = 1)

# Pre Processing
stop_words = stopwords.words('english')
wnl = WordNetLemmatizer()

def preprocess(text_column):
       """
       To change the sentence in the text variable by lemmatizing and removing
       the stop words

       Parameters
       ----------
       text_column : string. the tweet itself

       Returns
       -------
       new_review : return a new column of tweets that have been refined

       """
       # Remove link,user and special characters
       # And Lemmatize the words
       new_review = []
       for review in text_column:
              # this text is a list of tokens for the review
              text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(review).lower()).strip()
              text = [wnl.lemmatize(i) for i in text.split(' ') if i not in stop_words]
              new_review.append(' '.join(text))
       return new_review

# create a keyword dataset
train_keywords = pd.DataFrame([i for i in train_keywords if type(i) == str]).drop_duplicates()
train_keywords = list(train_keywords.iloc[:,0])
for i in range(len(train_keywords)):
       if '%20' in train_keywords[i]:
              new = re.sub('%20', ' ', train_keywords[i])
              new_indiv = new.split(' ')
              train_keywords[i] = new
              train_keywords += new_indiv
train_keywords = [wnl.lemmatize(i) for i in train_keywords]

def clean_data(d):
       d['text'] = preprocess(d['text'])
       location = []
       keyword = []
       for i in range(len(d)):
              if type(d.loc[i, 'location']) == str:
                     location.append(1)
              else:
                     location.append(0)
              keyword.append(sum([j in train_keywords for j in d.loc[i, 'text'].split(' ')]))
       d['location'] = location
       d['keyword'] = keyword
       return d
       
train_data = clean_data(train_data)
train_x = train_data.loc[:, ['keyword', 'location', 'text']]
train_y = train_data.loc[:, 'target']
test_data = clean_data(test_data)
test_x = test_data.loc[:, ['keyword', 'location', 'text']]

cv = CountVectorizer()
cv.fit(train_x['text'])
train_x = np.append(cv.transform(train_x['text']).toarray().T, np.array(train_x.iloc[:, 0:2]).T, 0).T
test_x = np.append(cv.transform(test_x['text']).toarray().T, np.array(test_x.iloc[:, 0:2]).T, 0).T

# XGBoost
train_xgb = xgb.DMatrix(train_x, train_y)
test_xgb = xgb.DMatrix(test_x)

params = {'eta': 0.45,
          'max_depth': 10,
          'objective': 'binary:logistic'}
xgb_model = xgb.train(params, train_xgb, num_boost_round = 10)
y_pred = xgb_model.predict(test_xgb)
y_pred = np.where(y_pred > 0.5, 1, 0)

final = pd.DataFrame(y_pred, test_data['id'])
final = final.reset_index()
final.columns = ['id', 'target']

final.to_csv('results.csv', index = False)
