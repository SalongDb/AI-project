#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd


# In[2]:


nltk.download('punkt')


# In[3]:


fake = pd.read_csv("Fake.csv")
genuine = pd.read_csv("True.csv")


# In[4]:


fake


# In[5]:


genuine


# In[6]:


fake["genuineness"] = 0
genuine["genuineness"] = 1


# In[7]:


data = pd.concat([fake, genuine], axis=0)


# In[8]:


data


# In[9]:


data = data.reset_index(drop= True)


# In[10]:


data = data.drop(['title','subject','date'],axis=1)


# # Data Preprocessing

# In[11]:


from nltk.tokenize import word_tokenize
data['text'] = data['text'].apply(word_tokenize)


# In[12]:


from nltk.stem.snowball import SnowballStemmer
sb = SnowballStemmer("english",ignore_stopwords=False)


# In[13]:


def stem_it(text):
    return [sb.stem(word) for word in text]


# In[14]:


data['text'] = data['text'].apply(stem_it)


# In[15]:


def stopword_removal(text):
    return [word for word in text if len(word)>>2]


# In[16]:


data['text'] = data['text'].apply(' '.join)


# In[17]:


data


# # Splitting data set

# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(data['text'],data['genuineness'], test_size=0.25)


# In[20]:


x_train


# # Vectorization (TFIDF)

# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.7)

tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)


# # Building of ML Model

# In[22]:


from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=900)
model1.fit(tfidf_train, y_train)


# In[23]:


pred1 = model1.predict(tfidf_test)


# In[24]:


pred1


# In[25]:


y_test


# In[26]:


from sklearn.metrics import accuracy_score
cr1 = accuracy_score(y_test, pred1)
cr1*100


# In[27]:


from sklearn.linear_model import PassiveAggressiveClassifier

model2 = PassiveAggressiveClassifier(max_iter=100)
model2.fit(tfidf_train, y_train)


# In[28]:


pred2 = model2.predict(tfidf_test)


# In[29]:


cr2 = accuracy_score(y_test, pred2)
cr2*100


# In[ ]:




