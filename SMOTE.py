#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
import sklearn.datasets as skds
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
# For reproducibility
np.random.seed(1237)


# In[3]:


#PREPROCESSING
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
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    df.loc[index,'text_prep'] = str(Final_words)


# In[4]:


Encoder = LabelEncoder()

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_prep'],df['Category'],test_size=0.3, random_state=13)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text_prep'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[5]:


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

sampling_strategy = {0: 1034, 1: 1034, 2: 1034}
#sm = SMOTE(sampling_strategy = sampling_strategy, random_state=6)
ros=RandomOverSampler(sampling_strategy=sampling_strategy, random_state=5)
x_train_res, y_train_res = ros.fit_resample(Train_X_Tfidf, Train_Y)


# In[6]:


import matplotlib.pyplot as plt
from collections import Counter

def plot_pie(y):
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
        return my_autopct

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True,
           autopct=make_autopct(sizes))
    ax.axis('equal')

print('Information of the Twitter dataset after oversampling: \n sampling_strategy={} \n y: {}'
      .format(sampling_strategy, Counter(y_train_res)))
plot_pie(y_train_res)


# In[7]:


Naive = naive_bayes.MultinomialNB()
Naive.fit(x_train_res, y_train_res)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print("Naive Bayes Precision Score -> ",precision_score(predictions_NB, Test_Y, average=None)*100)
print("Naive Bayes Recall Score -> ",recall_score(predictions_NB, Test_Y, average=None)*100)
print("Naive Bayes F1 Score -> ",f1_score(predictions_NB, Test_Y, average=None)*100)


# In[8]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', random_state=12)
SVM.fit(x_train_res, y_train_res)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("SVM Precision Score -> ",precision_score(predictions_SVM, Test_Y, average=None)*100)
print("SVM Recall Score -> ",recall_score(predictions_SVM, Test_Y, average=None)*100)
print("SVM F1 Score -> ",f1_score(predictions_SVM, Test_Y, average=None)*100)


# In[9]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(predictions_NB, Test_Y)
cm


# In[10]:


from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced', np.unique(Train_Y), Train_Y)
num_epochs = 10
batch_size=128

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text_prep'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[12]:


Encoder = LabelEncoder()
Y = Encoder.fit_transform(df['Category'])


# In[14]:


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
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[15]:


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
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[16]:


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=25, random_state=1)
predictions_RF=clf_rf.fit(Train_X_Tfidf, Train_Y).predict(Test_X_Tfidf)
print("RF Accuracy Score -> ", accuracy_score(predictions_RF, Test_Y)*100)
print("RF Precision Score -> ",precision_score(predictions_RF, Test_Y, average=None)*100)
print("RF Recall Score -> ",recall_score(predictions_RF, Test_Y, average=None)*100)
print("RF F1 Score -> ",f1_score(predictions_RF, Test_Y, average=None)*100)


# In[229]:


import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


y_true = Test_Y
y_pred = predictions_RF
labels = np.unique(Train_Y)

cm = confusion_matrix(y_true, y_pred, labels=labels)

cmap=plt.cm.Blues
plt.imshow(cm, interpolation='nearest', cmap=cmap)
#plt.title('Confusion matrix - number of predictions')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], '.2f'),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[17]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from inspect import signature

# Use label_binarize to be multi-label like settings
Y = label_binarize(Y, classes=[0, 1, 2])
n_classes = Y.shape[1]

# Split into training and test
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df['text_prep'], Y, test_size=.3,
                                                    random_state=10)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text_prep'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
#classifier = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=12))
classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=25, random_state=1))
#classifier = OneVsRestClassifier(naive_bayes.MultinomialNB())

classifier.fit(Train_X_Tfidf, Y_train)
y_score = classifier.predict_proba(Test_X_Tfidf)
#y_score = classifier.decision_function(Test_X_Tfidf)

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))


# In[18]:


plt.figure()
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


# In[19]:


from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='grey', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class model SMOTE - RF')
plt.legend(lines, labels, loc=(0, -.45), prop=dict(size=14))
plt.savefig('sample4RF.png')

plt.show()

