#!/usr/bin/env python
# coding: utf-8

# In[22]:


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


# In[23]:


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


# In[24]:


df.head()


# In[25]:


Encoder = LabelEncoder()
Y = Encoder.fit_transform(df['Category'])

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['text_prep'], Y,random_state=12, test_size=0.3)


# In[26]:


Tfidf_vect = TfidfVectorizer(max_features=None)
Tfidf_vect.fit(df['text_prep'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
print( Train_X_Tfidf.shape)
print(Test_X_Tfidf.shape)


# In[27]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print("Naive Bayes Precision Score -> ",precision_score(predictions_NB, Test_Y, average='weighted')*100)
print("Nayve Bayes Recall Score -> ",recall_score(predictions_NB, Test_Y, average='weighted')*100)
print("Naive Bayes F1 Score -> ",f1_score(predictions_NB, Test_Y, average='weighted')*100)


# In[28]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1, kernel='linear', probability=True, random_state=12)
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("SVM Precision Score -> ",precision_score(predictions_SVM, Test_Y, average=None)*100)
print("SVM Recall Score -> ",recall_score(predictions_SVM, Test_Y, average=None)*100)
print("SVM F1 Score -> ",f1_score(predictions_SVM, Test_Y, average='weighted')*100)


# In[29]:


import matplotlib.pyplot as plt

# get the probability distribution
probas = SVM.predict_proba(Test_X_Tfidf)
categ = np.unique(df['Category'])
# plot
plt.figure(dpi=90)
plt.hist(probas, bins=20)
plt.title('Classification Probabilities')
plt.xlabel('Probability')
plt.ylabel('# of Instances')
plt.xlim([0.5, 1.0])
plt.legend(categ)
plt.show()


# In[30]:


#now let's try with the bag of words feature extraction model
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features=3400)
                            
X_train_bow = count_vect.fit_transform(Train_X)
X_test_bow = count_vect.fit_transform(Test_X)
#change to array
X_train_bow = X_train_bow.toarray()
X_test_bow = X_test_bow.toarray()
print( X_train_bow.shape)
print(Train_Y.shape)
print(X_test_bow.shape)
print(Test_Y.shape)


# In[10]:


Naive = naive_bayes.MultinomialNB()
Naive.fit(X_train_bow,Train_Y)
# predict the labels on validation dataset
predictions_NB_bow = Naive.predict(X_test_bow)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB_bow, Test_Y)*100)


# In[11]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear')
SVM.fit(X_train_bow,Train_Y)
# predict the labels on validation dataset
predictions_SVM_bow = SVM.predict(X_test_bow)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM_bow, Test_Y)*100)


# In[31]:


#ALTERNATIVE USING PIPELINE - NB
from sklearn.pipeline import Pipeline
text_class = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', naive_bayes.MultinomialNB()),])

text_clf = text_class.fit(Train_X, Train_Y)


# In[32]:


predicted = text_clf.predict(Test_X)
np.mean(predicted == Test_Y)*100


# In[16]:


#ALTERNATIVE USING PIPELINE - SVM
from sklearn.pipeline import Pipeline
text_class = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', svm.SVC()),])

text_clf = text_class.fit(Train_X, Train_Y)

predicted = text_clf.predict(Test_X)
np.mean(predicted == Test_Y)*100


# In[33]:


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=25, random_state=1,class_weight= "balanced")
predictions_RF=clf_rf.fit(Train_X_Tfidf, Train_Y).predict(Test_X_Tfidf)
print("RF Accuracy Score -> ", accuracy_score(predictions_RF, Test_Y)*100)
print("RF Precision Score -> ",precision_score(predictions_RF, Test_Y, average=None)*100)
print("RF Recall Score -> ",recall_score(predictions_RF, Test_Y, average=None)*100)
print("RF F1 Score -> ",f1_score(predictions_RF, Test_Y, average=None)*100)


# In[34]:


from sklearn.pipeline import Pipeline
text_class = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', RandomForestClassifier()),])

text_clf = text_class.fit(Train_X, Train_Y)

predicted = text_clf.predict(Test_X)
np.mean(predicted == Test_Y)*100


# In[564]:


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


from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
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
#classifier = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=10))
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


# In[20]:


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


# In[21]:


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
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
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
plt.title('Extension of Precision-Recall curve to multi-class model - RF lemma/no stops')
plt.legend(lines, labels, loc=(0, -.45), prop=dict(size=14))

plt.show()


# In[599]:


import matplotlib.pyplot as plt
import numpy as np
import itertools

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
        color="white" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[600]:


import matplotlib.pyplot as plt
import numpy as np
import itertools

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

