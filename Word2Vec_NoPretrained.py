#!/usr/bin/env python
# coding: utf-8

# In[167]:


import re
import pandas as pd
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from gensim.models.phrases import Phrases, Phraser
import multiprocessing
from gensim.models import Word2Vec
import spacy


df = pd.read_csv('/Users/lucrezialamanna/Desktop/preprocessed_data.csv')
df = df[pd.notnull(df['Text'])]
df.to_csv('/Users/lucrezialamanna/Desktop/preprocessed_data.csv')

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
doc = open('/Users/lucrezialamanna/Desktop/preprocessed_data.csv').read()

def cleaning(doc):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    #if len(txt) > 2:
    return ' '.join(txt)
    
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['Text'])
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

text = pd.DataFrame(txt)
text.columns = ['Clean']


# In[168]:


df['Text'].shape


# In[169]:


#df.isnull().values.any()
#text = text.dropna()
text.shape
#text.isnull().values.any()
#text[pd.isna(text['Clean'])]


# In[170]:


sent = [row.split() for row in text['Clean']]
phrases = Phrases(sent, min_count=15, progress_per=500)
bigram=Phraser(phrases)
sentences = bigram[sent]


# In[101]:


text.shape


# In[102]:


cores = multiprocessing.cpu_count() # Count the number of cores in a computer

w2v_model = Word2Vec(min_count=10,
                     window=10,
                     size=200,
                     sample=1e-3, 
                     alpha=0.03, 
                     min_alpha=0.0001, 
                     negative=10,
                     workers=cores-1, 
                     sg=1)


# In[103]:


df['Category'].shape


# In[104]:


t = time()

w2v_model.build_vocab(sentences, progress_per=500)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))


# In[105]:


t = time()

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))


# In[106]:


w2v_model.wv.most_similar(positive=["corbyn"])


# In[160]:


from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

word_vectors=w2v_model.wv
fname = get_tmpfile("vectors.kv")
word_vectors.save(fname)
word_vectors = KeyedVectors.load(fname)
word_vectors.save_word2vec_format('vec_file.txt')


# In[ ]:


#RETROFITTING - run script now
w2v_model = KeyedVectors.load_word2vec_format("/Users/lucrezialamanna/Desktop/out_vec2.txt", binary=False)


# In[165]:


w2v_model.wv.most_similar(positive=["immigrant"])


# In[166]:


class MeanEmbeddingVectorizer(object):
    
    def __init__(self, w2v_model):
        self.w2v_model = w2v_model
        self_vector_size = w2v_model.wv.vector_size
        
    def fit(self):
        return self
    
    def transform(self, docs):
        doc_word_vector = self.word_average_list(docs)
        return doc_word_vector


    def word_average(self, sent):
        mean = []
        for word in sent:
            if word in self.w2v_model.wv.vocab:
                mean.append(self.w2v_model.wv.get_vector(word))

        if not mean:
            logging.warning("cannot compute average owing to no vector for {}".format(sent))
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean


    def word_average_list(self, docs):
        return np.vstack([self.word_average(sent) for sent in docs])


# In[122]:


#from UtilWordEmbedding import MeanEmbeddingVectorizer
import numpy as np

mean_vec_tr = MeanEmbeddingVectorizer(w2v_model)
doc_vec = mean_vec_tr.transform(sentences)


# In[123]:


from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

Encoder = LabelEncoder()
Y = Encoder.fit_transform(df['Category'])

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(doc_vec, Y, random_state=5, test_size=0.3, stratify=df['Category'])


# In[142]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', random_state=17)
SVM.fit(Train_X, Train_Y)
# predict the labels on validation dataset
predictions_SVM_ME = SVM.predict(Test_X)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM_ME, Test_Y)*100)
print("SVM Precision Score -> ",precision_score(predictions_SVM_ME, Test_Y, average=None)*100)
print("SVM Recall Score -> ",recall_score(predictions_SVM_ME, Test_Y, average=None)*100)
print("F1 Recall Score -> ",f1_score(predictions_SVM_ME, Test_Y, average=None)*100)


# In[143]:


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=25, random_state=18,class_weight= "balanced")
predictions_RF_ME=clf_rf.fit(Train_X, Train_Y).predict(Test_X)
#print("RF Accuracy Score -> ", clf_rf.score(Test_X_Tfidf, Test_Y)*100)
print("RF Accuracy Score -> ", accuracy_score(predictions_RF_ME, Test_Y)*100)
print("RF Precision Score -> ",precision_score(predictions_RF_ME, Test_Y, average=None)*100)
print("RF Recall Score -> ",recall_score(predictions_RF_ME, Test_Y, average=None)*100)
print("RF F1 Score -> ",f1_score(predictions_RF_ME, Test_Y, average=None)*100)


# In[144]:


import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


y_true = Test_Y
y_pred = predictions_SVM_ME
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


# In[146]:


import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


y_true = Test_Y
y_pred = predictions_RF_ME
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
        color="red" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[147]:


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        #self.dim = len(word2vec.values().next())
        self_vector_size = w2v_model.wv.vector_size

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self_vector_size)], axis=0)
                for words in X
            ])


# In[148]:


tfidf_vec_tr = TfidfEmbeddingVectorizer(w2v_model)
tfidf_vec_tr.fit(sentences, df['Category'])  # fit tfidf model first
tfidf_doc_vec = tfidf_vec_tr.transform(sentences)


# In[149]:


from sklearn import model_selection

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(tfidf_doc_vec, df['Category'], random_state=4, test_size=0.3, stratify=df['Category'])

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', random_state=12)
SVM.fit(Train_X, Train_Y)
# predict the labels on validation dataset
predictions_SVM_TF = SVM.predict(Test_X)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM_TF, Test_Y)*100)
print("SVM Precision Score -> ",precision_score(predictions_SVM_TF, Test_Y, average=None)*100)
print("SVM Recall Score -> ",recall_score(predictions_SVM_TF, Test_Y, average=None)*100)
print("F1 Recall Score -> ",f1_score(predictions_SVM_TF, Test_Y, average=None)*100)


# In[150]:


import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


y_true = Test_Y
y_pred = predictions_SVM_TF
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


# In[151]:


from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=25, random_state=18,class_weight= "balanced")
predictions_RF=clf_rf.fit(Train_X, Train_Y).predict(Test_X)
#print("RF Accuracy Score -> ", clf_rf.score(Test_X_Tfidf, Test_Y)*100)
print("RF Accuracy Score -> ", accuracy_score(predictions_RF, Test_Y)*100)
print("RF Precision Score -> ",precision_score(predictions_RF, Test_Y, average=None)*100)
print("RF Recall Score -> ",recall_score(predictions_RF, Test_Y, average=None)*100)
print("RF F1 Score -> ",f1_score(predictions_RF, Test_Y, average=None)*100)


# In[152]:


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
        color="red" if cm[i, j] > thresh else "black"
    )

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()


# In[157]:


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
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(doc_vec, Y, test_size=.3,
                                                    random_state=10)


# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier

# Run classifier
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', random_state=12))
#classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=25, random_state=1))


classifier.fit(Train_X, Y_train)
#y_score = classifier.predict_proba(Test_X)
y_score = classifier.decision_function(Test_X)

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


# In[158]:


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


# In[159]:


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
plt.title('Extension of Precision-Recall curve to multi-class model - SVM -no pre-trained')
plt.legend(lines, labels, loc=(0, -.45), prop=dict(size=14))
plt.savefig('sample4RF.png')

