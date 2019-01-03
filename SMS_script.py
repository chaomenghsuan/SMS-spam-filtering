#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import pandas as pd
import string
import re
import itertools as it
import nltk
from sklearn.naive_bayes import MultinomialNB
import sklearn
from sklearn import svm
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import csr_matrix

# structural feature parameters

feat = [True, True, True, True, True, True]
inDir = '/Users/zhaomengxuan/Documents/text mining/final project/spam.csv'

######################
## data extraction  ##
######################
print('='*10+'\ndata loading')
# Raw data processing: extract data from .csv file

sms = pd.read_csv(inDir, encoding='latin-1')

#Drop column and name change
sms = sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
sms = sms.rename(columns={"v1":"label", "v2":"text"})

#Count observations in each label
print(sms.label.value_counts())

######################
# structural features #
######################
print('='*10+'\nfeatures extracting')
rawtext = [sms['text'][n] for n in range(len(sms))]

# structural feature 1: sms message length
sms_length = [len(st) for st in rawtext]

# structural feature 2: numeric character ratio
numeric = [len(re.findall(r'\d', rawtext[i]))/len(rawtext[i]) for i in range(len(rawtext))]

# structural features 3: non-alphanumeric character ratio
non_alphanumeric_pt = re.compile(r'[^\w\s]+')
non_alphanumeric = [len(''.join(re.findall(non_alphanumeric_pt, rawtext[i])))/len(rawtext[i]) for i in range(len(rawtext))]

# structural feature 4: if sms message include url
# include: 1, non: 0
url_index = [rawtext.index(st) for st in rawtext if 'http://' in st]
url = [0] * len(rawtext)
for i in url_index:
    url[i] += 1
assert sum(url) == len(url_index)

# structural feature 5: uppercase charactor ratio
text = [sent.translate(str.maketrans('', '', string.punctuation)) for sent in rawtext]
digits = re.compile(r'\b\d{10,}\b')
text = [re.sub(digits, '<longdigit>', sent) for sent in text]
def countupper(string):
    n = 0
    for letter in string:
        if letter.isupper(): n += 1
    ratio = n/len(string)
    return ratio
upper = [countupper(st) for st in text]

# structural reature 6: number of terms
text_lower = [[w.lower() for w in sent.split()] for sent in text]
terms_count = [len(t) for t in text_lower]

structural_all = [sms_length, numeric, non_alphanumeric, url, upper, terms_count]
structural_names = ['message length', 'numeric ratio', 'non_alphanumeric ratio', 'url existance', 'uppercase character ratio', 'terms count']
assert len(structural_all) == len(feat)
structural = []
for i in range(len(feat)):
	if feat[i]:
		structural.append(structural_all[i])
		print('feature included:', structural_names[i])
assert len(structural) == sum(feat)
if len(structural):
	structural = np.array(structural).T
print('number of structural features adopted', len(feat))

#######################
# train/test spliting #
#######################

tr = int(len(text_lower)*0.7)
dev = int(len(text_lower)*0.8)
text_train, text_dev, text_test = text_lower[:tr], text_lower[tr:dev], text_lower[dev:]

text_train, text_test, text_dev = \
[' '.join(sent) for sent in text_train],\
[' '.join(sent) for sent in text_test],\
[' '.join(sent) for sent in text_dev]

vec = CountVectorizer()
vec.fit(text_train)
Xtr, Xte, Xdev = vec.transform(text_train), \
vec.transform(text_test), vec.transform(text_dev)

Xtr, Xte, Xdev = \
csr_matrix(Xtr).toarray(), csr_matrix(Xte).toarray(), csr_matrix(Xdev).toarray()

if len(structural):
	structural_train, structural_dev, structural_test = \
	structural[:tr], structural[tr:dev], structural[dev:]

	Xtr, Xte, Xdev = \
	np.concatenate((Xtr.T, structural_train.T)).T, \
	np.concatenate((Xte.T, structural_test.T)).T, \
	np.concatenate((Xdev.T, structural_dev.T)).T


print('training data:', Xtr.shape[0], 'developing:', Xdev.shape[0], 'testing:', Xte.shape[0])

#######################
#######  labels  ######
#######################

# Labels for NB/SVM: 0 for non-spam, 1 for spam
simplelabels = []
for i in range(len(sms)):
    if sms['label'][i] == 'ham':
        simplelabels.append(0)
    elif sms['label'][i] == 'spam':
        simplelabels.append(1)
simplelabels = np.array(simplelabels)

# Labels for tensorflow: [0,1] for non-spam, [1,0] for spam
labels = []
for i in range(len(sms)):
    if sms['label'][i] == 'ham':
        labels.append([1,0])
    elif sms['label'][i] == 'spam':
        labels.append([0,1])
labels = np.array(labels)

# label spliting
simpytr, simpyte = simplelabels[:tr], simplelabels[dev:]
ytr, ydev, yte = labels[:tr], labels[tr:dev], labels[dev:]


#######################
####  Naive Bayes  ####
#######################

print('='*10+'\nNaive Bayes Processing')
clf_nb = MultinomialNB()
clf_nb.fit(Xtr, simpytr)
pred_nb = clf_nb.predict(Xte)

print('='*10+'\nNaive Bayes Result')
print('Precission:', round(sklearn.metrics.precision_score(simpyte, pred_nb), 3))
print('Recall:', round(sklearn.metrics.recall_score(simpyte, pred_nb), 3))
print('F1 Score:', round(sklearn.metrics.f1_score(simpyte, pred_nb), 3))
print(pd.DataFrame(sklearn.metrics.confusion_matrix(simpyte, pred_nb), index=['non-spam', 'spam'], 
             columns=['predict non-spam','predict spam']))

#######################
#######  K-NN  ########
#######################

print('='*10+'\nK-NN Processing')
clf_knn = KNeighborsClassifier(n_neighbors = 1)
clf_knn.fit(Xtr, simpytr)
pred_knn = clf_knn.predict(Xte)

print('='*10+'\nK-NN Result')
print('Precission:', round(sklearn.metrics.precision_score(simpyte, pred_knn), 3))
print('Recall:', round(sklearn.metrics.recall_score(simpyte, pred_knn), 3))
print('F1 Score:', round(sklearn.metrics.f1_score(simpyte, pred_knn), 3))
print(pd.DataFrame(sklearn.metrics.confusion_matrix(simpyte, pred_knn), index=['non-spam', 'spam'], 
             columns=['predict non-spam','predict spam']))

#######################
########  SVM  ########
#######################

print('='*10+'\nSVM Processing')
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(Xtr, simpytr)
pred_svm = clf_svm.predict(Xte)

print('='*10+'\nSVM Result')
print('Precission:', round(sklearn.metrics.precision_score(simpyte, pred_svm), 3))
print('Recall:', round(sklearn.metrics.recall_score(simpyte, pred_svm), 3))
print('F1 Score:', round(sklearn.metrics.f1_score(simpyte, pred_svm), 3))
print(pd.DataFrame(sklearn.metrics.confusion_matrix(simpyte, pred_svm), index=['non-spam', 'spam'], 
             columns=['predict non-spam','predict spam']))

#######################
####  Neural Net  #####
#######################

print('='*10+'\nNeural Network Processing')
num_input = Xtr.shape[1]
num_classes = ytr.shape[1]
# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

n_hidden_1 = 8
n_hidden_2 = 8
n_hidden_3 = 8

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 8 neurons
    layer_1 = tf.nn.sigmoid((tf.add(tf.matmul(x, weights['h1']), biases['b1'])))
    # Hidden fully connected layer with 8 neurons
    layer_2 = tf.nn.sigmoid((tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])))
    # Hidden fully connected layer with 8 neurons
    layer_3 = tf.nn.sigmoid((tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.sigmoid((tf.matmul(layer_3, weights['out']) + biases['out']))
    return out_layer

 # Parameters
learning_rate = 0.1
num_steps = 300
display_step = 50

# Construct model
logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.nn.l2_loss(logits - Y)
optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: Xtr, Y: ytr})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: Xtr,
                                                                 Y: ytr})
            print("Step " + str(step) + ", L2 Loss= " + \
                  "{:.3f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # calculate precision, recall and f1 score on dev set, using sklearn
    pred_label = sess.run(logits, feed_dict={X: Xdev})
    pred = sess.run(tf.argmax(pred_label,1))
    y = sess.run(tf.argmax(ydev,1))
    
    print('\nDevelopment set results:')
    print('precision:', round(sklearn.metrics.precision_score(y, pred),3))
    print('recall:', round(sklearn.metrics.recall_score(y, pred),3))
    print('F1 score:',round(sklearn.metrics.f1_score(y, pred),3))
    print(pd.DataFrame(sklearn.metrics.confusion_matrix(y, pred), index=['non-spam', 'spam'], 
             columns=['predict non-spam','predict spam']))
    
    # calculate precision, recall and f1 score on testing set 
    pred_label_te = sess.run(logits, feed_dict={X: Xte})
    pred_te = sess.run(tf.argmax(pred_label_te,1))
    y_te = sess.run(tf.argmax(yte,1))
    
    print('\nTesting set results:')
    print('precision:', round(sklearn.metrics.precision_score(y_te, pred_te),3))
    print('recall:', round(sklearn.metrics.recall_score(y_te, pred_te),3))
    print('F1 score:',round(sklearn.metrics.f1_score(y_te, pred_te),3))
    print(pd.DataFrame(sklearn.metrics.confusion_matrix(y_te, pred_te), index=['non-spam', 'spam'], 
             columns=['predict non-spam','predict spam']))


