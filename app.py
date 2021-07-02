from operator import mod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import nltk
import re
import string
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import tweepy as tw
import random

# initialising the variables
count_vect = CountVectorizer(stop_words='english')
urls = ["https://www.talkspace.com/blog/category/depression/",
        "https://www.pickthebrain.com/blog/category/depression/",
        "https://www.time-to-change.org.uk/category/blog/depression",
        "http://chipur.com/",
        "https://beyondmeds.com/",
        "http://daisiesandbruises.com/",
        "http://thesplinteredmind.blogspot.com/"]

# loading the data
@st.cache(allow_output_mutation=True)
def load_data():
  df = pd.read_csv('sentiments.csv')
  return df

# defining a function to remove slang text from the tweets
slang_words = { 'lol': 'laugh out loud', 'dm': 'direct message', 'prolly': 'probably', 'aint': 'are not','rt': 'retweet', 'btw': 'by the way', 'tbh': 'to be honest', 'diy': 'do it yourself', 'hbd': 'happy birth day', 'hmu': 'hit me up', 'mcm': 'man crush monday', 'wcw': 'woman crush wednesday'}
def removeSlangText(text):
  for word in text.split():
    if word in slang_words.keys():
      text = text.replace(word, slang_words[word])
  return text

# cleaning the tweets
PUNCT_TO_REMOVE = string.punctuation
def cleanTweets(text):
  # lower casing all text
  text = text.lower()
  # removing all url links
  text = re.sub(r'http\S+', '', text)
  # removing twitter mentions
  text = re.sub(r'@[\w]+', '', text)
  # removing the retweet tag
  text = re.sub(r'rt\S+','', text)
  # removing all other punctuations
  text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
  # returning the text
  return text

# defining the function for removing stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
  return " ".join([word for word in str(text).split(' ') if word not in STOPWORDS])

# downloading the necessary packages for lemmatization
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
# lemmatizing the tweets
lemmatizer = nltk.WordNetLemmatizer()
def lemmatizeTweets(text):
    text = lemmatizer.lemmatize(text)
    return text

@st.cache(persist=True)
def preprocess(df):
  df['Tweets'].replace(r'[0-9]','',regex=True, inplace = True)
  # cleaning the 'Tweets' column
  df['Tweets'] = df['Tweets'].apply(lambda tweet: remove_stopwords(tweet))
  df['Tweets'] = df['Tweets'].apply(lambda tweet: cleanTweets(tweet))
  df['Tweets'] = df['Tweets'].apply(lambda tweet: lemmatizeTweets(tweet))
  # vectorising the data
  X = df['Tweets']
  y = df['Label']
  return X,y

@st.cache(persist=True,allow_output_mutation=True)
def model_train(X,y):
  vectorizer = CountVectorizer(list(X),stop_words='english')
  features = vectorizer.fit_transform(X)
  # splitting the data into train set and test set
  X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)
  # training the model
  rnd_clf = RandomForestClassifier()
  rnd_clf.fit(X_train,y_train)
  pred = rnd_clf.predict(X_test)
  acc = metrics.accuracy_score(pred,y_test)
  # saving the model
  pickle_in = open('classifier.pkl','wb')
  clf_model = pickle.dump(rnd_clf, pickle_in)
  pickle_in.close()
  return acc, vectorizer

# twitter data mine
consumer_key= "2G7fC17Yl1DqtJZZhK9Hv5XXi"
consumer_secret = 'VVVDAKupyXJPslDebSEdtAFoJHUMRPcCnR9GPumL2MirdE04hD'
access_token = '87639892-wvdm1WFail9C4HVBnVDizdloPEw4HurpxDVsN7Vrx'
access_token_secret = 'WtfkfVVbTWSc9qNPZNk4jvIloaVRoQ2s7IkdmkY4997Xt'

def mineData(userID,count):
  tweetList = []
  auth = tw.OAuthHandler(consumer_key,consumer_secret)
  auth.set_access_token(access_token,access_token_secret)
  api = tw.API(auth)
  #getting the tweets
  tweets = api.user_timeline(screen_name=userID, count=count, include_rts=False, tweet_mode='extended')
  for tweet in tweets:
    tweetList.append(tweet.full_text)
  return tweetList



def main():
  # loading the data
  data = load_data()
  X,y = preprocess(data)
  acc, vectorizer = model_train(X,y)
  # loading the pretrained model
  pickle_out = open('classifier.pkl', 'rb')
  model = pickle.load(pickle_out)
  pickle_out.close()

  st.title("\tSocial Media Depression Detection")
  st.write(" ")
  st.write(" ")
  st.sidebar.image('sentiments.jpg')
  st.sidebar.header("Project Engineers")
  st.sidebar.text('1.Leone Dimairo')
  st.sidebar.text('2.Gift Takudzwa Chipungu')
  st.sidebar.text('3.Tinashe Calvin Makara')
    
  # designing the main screen
  if (st.checkbox('Custom Tweet')):
    tweet = st.text_input('Tweet Content')
    if (st.button('Submit')):
      tweet = pd.Series(tweet)
      # cleaning the tweet
      tweet.replace(r'[0-9]','',regex=True, inplace = True)
      tweet = tweet.apply(lambda text: remove_stopwords(text))
      tweet = tweet.apply(lambda text: cleanTweets(text))
      tweet = tweet.apply(lambda text: lemmatizeTweets(text))
      # making the prediction
      vector = vectorizer.transform(tweet)
      pred = model.predict(vector)
      # print out the prediction result
      st.write('Prediction result is ', pred[0])
      if (pred[0] == 1):
        st.success('Want to get help? Get genuine tips for fighting depression from the following blog site')
        st.write(random.choice(urls))
        
  if (st.checkbox('Predict Live Tweets')):
    handle = st.text_input('Twitter Username')
    count = st.number_input('Tweet Count')
    if (st.button('Predict')):
      tweetList = mineData(handle, count)
      twtList = np.array(tweetList)
      tweetList = pd.Series(tweetList)
      # cleaning the tweets
      tweetList.replace(r'[0-9]','',regex=True, inplace = True)
      tweetList = tweetList.apply(lambda tweet: remove_stopwords(tweet))
      tweetList = tweetList.apply(lambda tweet: cleanTweets(tweet))
      tweetList = tweetList.apply(lambda tweet: lemmatizeTweets(tweet))
      # making the predictions
      vectors = vectorizer.transform(tweetList)
      predictions = model.predict(vectors)
      # defining the dataframe to display
      df = pd.DataFrame({'Tweet': twtList, 'Prediction': predictions})
      st.write(df)
      result = int(df['Prediction'].mean().round(0))
      if (result == 0.0):
        st.write('Prediction result is : ', result)
      else:
        st.success('Want to get help? Get genuine tips for fighting depression from the following blog site')
        st.write(random.choice(urls))

if __name__ == "__main__":
    main()

