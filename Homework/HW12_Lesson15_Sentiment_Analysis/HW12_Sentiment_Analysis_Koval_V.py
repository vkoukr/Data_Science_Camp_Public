#===='Sentiment Analysis'
import pandas as pd
import numpy as np

#---Read Negative from file
fn='./data/rt-polarity.neg'
with open(fn, "r",encoding='utf-8', errors='ignore') as f: # some invalid symbols encountered 
    content = f.read()  
texts_neg=  content.splitlines()
print ('len of texts_neg = {:,}'.format (len(texts_neg)))
for review in texts_neg[:5]:
    print(f'===== Length of negative review is: {len(review)}')
    print ( '\n\n', review)
df_neg=pd.DataFrame(texts_neg, columns=['reviews_text'])
df_neg['Rating_binary'] = 0
    
#---Read Positive from file
fn='./data/rt-polarity.pos'

with open(fn, "r",encoding='utf-8', errors='ignore') as f:
    content = f.read()
texts_pos=  content.splitlines()
print ('len of texts_pos = {:,}'.format (len(texts_pos)))
for review in texts_pos[:5]:
    print(f'===== Length of positive review is: {len(review)}')
    print ('\n', review)

df_pos=pd.DataFrame(texts_pos, columns=['reviews_text'])
df_pos['Rating_binary'] = 1

# fn='./data/demonetization-tweets.csv'
# df = pd.read_csv(fn, encoding="latin-1")
# df.dropna(inplace=True) 

df=pd.concat([df_pos, df_neg], ignore_index=True)

# ===== Using NaiveBase Classifyer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
import random

#--- Tokenize texts
def preprocess(text): # removes punctualtion
    tokenizer = RegexpTokenizer(r'\w+') # just for demo 
    return tokenizer.tokenize(text.lower())

pos_words = [preprocess(str(texts_pos[i])) for i in range(len(texts_pos))]
print (len(pos_words))
print(pos_words[:100])

neg_words = [preprocess(str(texts_neg[i])) for i in range(len(texts_neg))]
print (len(neg_words))
print(neg_words[:100])

# # --- Build vocabulary with most frequent words
all_words=preprocess(str(texts_pos+texts_neg))
all_words=nltk.FreqDist(all_words)
print ('len of vocabulary: {:,}'.format (len(all_words)))

most_common_words = list(zip(*all_words.most_common()))[0] # [0] means names whereas [1] are frequencies 
# most_common(5000) - it may retutn limited number but in this sample the features will be filtered later after removing stop words 
print (most_common_words[:100])

# ----- Remove stop words
def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))  
    return [w for w in words if w not in stop_words]

most_common_words_filtered = remove_stop_words(most_common_words)
print(most_common_words_filtered[:100]) 

pos_words=[remove_stop_words(pos_words[i]) for i in range(len(pos_words))] 
neg_words=[remove_stop_words(neg_words[i]) for i in range(len(neg_words))] 

#---Select Features
word_features = most_common_words_filtered[:3000]
print (word_features[:100])

# documents = [(pos_words, 'pos')]
documents = [(pos_words[i], 'pos') for i in range(len(pos_words))]

# documents = documents + [(neg_words, 'neg')]
documents = documents + [(neg_words[i], 'neg') for i in range(len(neg_words))]

# ----shuffle first 
random.shuffle(documents) # it is inplace method
documents= documents[:500] # reduce the data set for speed up the demo 
len (documents)

# ---- Vectorize documents
def find_features(review_tokens):
    return {w: w in set(review_tokens) for w in word_features} # feature representation on document

data_set= [(find_features(review_tokens), category) for (review_tokens, category) in documents]
print(data_set[0])

#--- Split to training and test set
split_on = int(len(data_set)*.8)
X_y_train= data_set[:split_on]
X_y_test = data_set[split_on:]
print (len(X_y_train))

# ---- Train model
from nltk.classify import NaiveBayesClassifier
clf= NaiveBayesClassifier.train(X_y_train)

# ------ Evaluate mode
print('\n\nThe Accuracy of tested NaiveBaseClassifyer is: ',nltk.classify.accuracy(clf, X_y_test)*100,'\n')

# ----- Review most informative features
print(clf.show_most_informative_features(5))

#---- Incorporate with sklearn
from nltk.classify.scikitlearn import SklearnClassifier # this is wrapper to incorporate with sklearn using nltk style.
from sklearn.naive_bayes import MultinomialNB

# Convert to nltk classifiers 
MNNB_classifier= SklearnClassifier(MultinomialNB()) # Note : use ()

from sklearn.linear_model import LogisticRegression
lr_classifier = SklearnClassifier(LogisticRegression()) 

from sklearn.svm import SVC, LinearSVC, NuSVC # NuSVC - Similar to SVC but uses a parameter to control the number of support vectors.
svc_clf = SklearnClassifier(SVC())  
lin_svc_clf= SklearnClassifier(LinearSVC())  
nu_svc_clf = SklearnClassifier(NuSVC())

# native nltk classifier
clf= nltk.NaiveBayesClassifier.train(X_y_train) 
print('Accuracy nltk.NaiveBayesClassifier={}%'.format(nltk.classify.accuracy(clf,X_y_test) * 100))
# clf.show_most_informative_features(15)
MNNB_classifier.train(X_y_train)
print('Accuracy MNNB_classifier ={}%'.format(nltk.classify.accuracy(MNNB_classifier, X_y_test) * 100)) # 79.0%
lr_classifier.train(X_y_train)
print('Accuracy lr_classifier ={}%'.format(nltk.classify.accuracy(lr_classifier, X_y_test) * 100)) # 82.0%
svc_clf.train(X_y_train)
print('Accuracy svc_clf={}%'.format(nltk.classify.accuracy(svc_clf, X_y_test) * 100)) # 52.0% - default is rbf kernel
lin_svc_clf.train(X_y_train)
print('Accuracy lin_svc_clf={}%'.format(nltk.classify.accuracy(lin_svc_clf, X_y_test) * 100)) # 82.0%
nu_svc_clf.train(X_y_train)
print('Accuracy nu_svc_clf={}%'.format(nltk.classify.accuracy(nu_svc_clf, X_y_test) * 100)) #

# ===== Using sklearn
print('\n\n =======SCKITLEARN NOW=======\n')
#---Spliting [Train/test]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['reviews_text'],df['Rating_binary'],random_state=0)
# Review training sample
print(X_train.iloc[0], y_train.iloc[0])

#-----Count vectorizer
from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer().fit(X_train) # Fit the CountVectorizer to the training data
print('features samples:\n{}'.format(vect.get_feature_names()[::2000])) # display each 2000-th feature 
print ('\nlen of features {:,}'.format(len(vect.get_feature_names()))) 

# ---Transfrom the X_train to feature representation
X_train_vectorized = vect.transform(X_train) # indeces of existing words from vocabulary and their count in current text
print (X_train_vectorized[0])

# --- Review vectorized training sample
df = pd.DataFrame(X_train_vectorized[0].toarray(), index= ['value']).T
df.head()

print (list(df[df['value']>0].index))
[vect.get_feature_names()[index] for index in df[df['value']>0].index.values]

# ---- Train model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

clf = LogisticRegression(max_iter=2000).fit(X_train_vectorized, y_train) # Train the model
# Evaluate model
predictions = clf.predict(vect.transform(X_test)) # Predict the transformed test documents
print('f1: ', f1_score(y_test, predictions)) 
scores = clf.decision_function(vect.transform(X_test)) 
print('AUC: ', roc_auc_score(y_test, scores)) 

# ---Review relevant features
# the smallest coefs corresponds to Neg impact, and largest coefs represent Pos impact

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = clf.coef_[0].argsort() # ascending  [0] is just squeeze from shape (1,n)
clf.coef_.shape, clf.coef_[0].shape, sorted(clf.coef_[0])[:10], sorted(clf.coef_[0])[-11:-1],

print('Smallest coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
# model.coef_[0][sorted_coef_index[0]] the smallest 



