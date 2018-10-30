
# coding: utf-8

# # apply LDA to a set of documents and split them into topics# 

# In[84]:


import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
from gensim import corpora, models


# ### DATA: Kaggle; https://www.kaggle.com/therohk/million-headlines/data

# In[134]:


dt = pd.read_csv('data/abcnews-date-text.csv', error_bad_lines=False)
data = dt[['headline_text']]
data['index']=data.index
data.head(5)


# # Data Preprocessing

# Tokenization: splitting the text (english) into words for a non-english language

# Lemmatization: reduce inflectional forms and sometimes derivationally related forms of a word to a common base form.

# In[3]:


def lemmatize_stemming(text):
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    #remove all stopword, words with less than 3 char
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


# test on the first headline 

# In[4]:


doc_sample = data[:1].values[0][0]
doc_sample
words=[]
for i in doc_sample.split(' '):
    words.append(i)
print(words)    
print (preprocess(doc_sample))


# In[5]:


processed_docs = data['headline_text'].map(preprocess)
print(processed_docs[:5])


# In[94]:


dct = gensim.corpora.Dictionary(processed_docs)


# We are filtering out words that appears is less than 15 docs, more than 0.5, keeping the 100000 first ones.

# In[95]:


bow_corpus = [dct.doc2bow(doc) for doc in processed_docs]


# Implementation of TF_IDF (text freq, inversed text freq)

# In[100]:


#fit model
tfidf = models.TfidfModel(bow_corpus)
#apply model 
vector = tfidf[bow_corpus]


# In[106]:


vector[0][::]


# In[108]:


lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dct, passes=2, workers=2)


# In[112]:


for idx, topic in lda_model.print_topics():
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[115]:


lda_model_tfidf = gensim.models.LdaMulticore(vector, num_topics=10, id2word=dct, passes=2, workers=4)


# In[116]:


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# Model evaluation

# In[125]:


processed_docs[0]
#using bag of word
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
#using LDA TF-IDF model
print('\n\n-----------------------------------------------------------\n\n')
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# In[132]:


unseen_document = 'new olympic champion get in jail'
bow_vector = dct.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

