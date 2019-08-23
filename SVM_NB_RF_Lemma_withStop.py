#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import spacy 
import re

np.random.seed(300)

df = pd.read_csv('/Users/lucrezialamanna/Desktop/preprocessed_data.csv')
df = df.astype(str)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#df = df.drop(['text_final'], axis=1)
df = df[pd.notnull(df['Category'])]
df.to_csv('/Users/lucrezialamanna/Desktop/preprocessed_data.csv')
df.head()


# In[3]:



# Step - a : Remove blank rows if any.
#df['Text'].dropna(inplace=True)# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
df['Text'] = [entry.lower() for entry in df['Text']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
df['Text']= [word_tokenize(entry) for entry in df['Text']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(df['Text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    df.loc[index,'text_prep'] = str(Final_words)


# In[4]:


df.head()


# In[15]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_prep'],df['Category'],random_state = 11, test_size=0.3)


# In[16]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# In[17]:


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text_prep'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[28]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print("Naive Bayes Precision Score -> ",precision_score(predictions_SVM, Test_Y, average='micro')*100)
print("Naive Bayes Recall Score -> ",recall_score(predictions_SVM, Test_Y, average='micro')*100)
print("Naive Bayes Recall Score -> ",f1_score(predictions_SVM, Test_Y, average='macro')*100)


# In[24]:


import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

y_true = Test_Y
y_pred = predictions_NB
labels = np.unique(Train_Y)

cm = confusion_matrix(y_true, y_pred, labels=labels)

cmap=plt.cm.Blues
plt.imshow(cm, interpolation='nearest', cmap=cmap)
#plt.title('Confusion matrix - number of predictions')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
        horizontalalignment="center",
        color="red" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[19]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[25]:


import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

y_true = Test_Y
y_pred = predictions_SVM
labels = np.unique(Train_Y)

cm = confusion_matrix(y_true, y_pred, labels=labels)

cmap=plt.cm.Blues
plt.imshow(cm, interpolation='nearest', cmap=cmap)
#plt.title('Confusion matrix - number of predictions')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
        horizontalalignment="center",
        color="red" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[20]:


#now let's try with the bag of words feature extraction model
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words="english", max_features=3000)
                            
X_train_bow = count_vect.fit_transform(Train_X)
X_test_bow = count_vect.fit_transform(Test_X)
#change to array
X_train_bow = X_train_bow.toarray()
X_test_bow = X_test_bow.toarray()
print( X_train_bow.shape)
print(Train_Y.shape)
print(X_test_bow.shape)
print(Test_Y.shape)


# In[21]:


Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train_bow,Train_Y)
# predict the labels on validation dataset
predictions_NB_bow = Naive.predict(X_test_bow)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)


# In[22]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train_bow,Train_Y)
# predict the labels on validation dataset
predictions_SVM_bow = SVM.predict(X_test_bow)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


# In[ ]:




