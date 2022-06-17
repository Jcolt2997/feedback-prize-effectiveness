#!/usr/bin/env python
# coding: utf-8

# In[27]:


#import necessary files
import numpy as np
import pandas as pd
import syllapy
from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
import warnings
import spacy
import matplotlib.pyplot as plt
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
pd.set_option('display.max_columns', 90)  # display all column of dataframe
pd.set_option('display.max_row', 100)     # display 100 rows of the dataframe
pd.set_option('display.max_colwidth', -1) # display all values within cells
pd.set_option('display.float_format', '{:.5f}'.format)
warnings.filterwarnings("ignore")


# In[2]:


def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


# This notebook will look at exploring the training and testing data for the [Feedback Prize - Predicting Effective Arguments](https://www.kaggle.com/competitions/feedback-prize-effectiveness/overview)

# In[3]:


df = pd.read_csv('/Users/jakelieberfarb/Desktop/college/Kaggle Competition/feedback-prize-effectiveness/Data/train.csv')
df.head()


# In[4]:


df.shape


# In[5]:


pie_data= count_values_in_column(df,"discourse_type") # save data aS a dataframe
count_values_in_column(df,"discourse_type")


# In[6]:


plt.figure(figsize=(10,10))
labels = ['Evidence', 'Claim', 'Position', 'Concluding Statement', 'Lead','Counterclaim', 'Rebuttal']
plt.title('Overview of "discourse_effectiveness" Data', fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[7]:


pie_data= count_values_in_column(df,"discourse_effectiveness") # save data aS a dataframe
count_values_in_column(df,"discourse_effectiveness")


# In[8]:


plt.figure(figsize=(10,10))
labels = ['Adequate','Effective', 'Ineffective']
plt.title('Overview of "discourse_effectiveness" Data', fontsize=30)
plt.pie(pie_data['Total'], labels=labels, autopct='%1.1f%%', textprops={'fontsize': 20})
plt.rcParams["axes.labelweight"] = "bold"
plt.tight_layout()
plt.show


# In[9]:


# label encode 'discourse_effectiveness'
le = LabelEncoder()
df['discourse_effectiveness_values'] = le.fit_transform(df.discourse_effectiveness.values)
df.tail()


# In[10]:


# Created functions for Removing Punctuation
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0–9]+', '', text)
    return text
df['punctuation'] = df['discourse_text'].apply(lambda x: remove_punct(x))

#Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text
df['tokenized'] = df['punctuation'].apply(lambda x: tokenization(x.lower()))

#remove last comma from tokenization
df['tokenized'] = [[s for s in l if s] for l in df['tokenized']]

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
df['nonstop'] = df['tokenized'].apply(lambda x: remove_stopwords(x))

#Appliyng Stemmer
ps = nltk.PorterStemmer()
def stemming(text):
    text = [ps.stem(word) for word in text]
    return text
df['stemmed'] = df['nonstop'].apply(lambda x: stemming(x))

print(df['tokenized'][1])
df.head(1)


# In[11]:


#create dummy variables for 'discourse_type'
y = pd.get_dummies(df.discourse_type)
df = df.join(y)
df.head(1)


# In[18]:


# count of the number of words in 'discourse_text'
df['word_count_discourse_text'] = df['discourse_text'].str.split().str.len()

df.head(2)


# In[45]:


# needs fixing, not getting the correct number of syllables
# get number of syllables 
syllables = []
for i in range(len(df['punctuation'])):
    x = syllapy.count(df['punctuation'][i])
    syllables.append(x)
pd.DataFrame(syllables)


# In[41]:


# other functions to make #

# average number of syllables per word for each 'discourse_text'

# syllable count of word with the most syllables for each 'discourse_text'
df.head(5)


# In[48]:


df.to_csv(path_or_buf='Data/' +'processed_data.csv',index=False)


# In[ ]:




