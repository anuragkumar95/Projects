# -*- coding: utf-8 -*-sad

# Author: Anurag Kumar

from nltk.corpus import nps_chat
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
from random import shuffle
import string
from spacy.lang.en import English
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords

#I am using the nps_chat corpus provided in NLTK module. I filtered data containing only 
#'Greet','Emotion','Statement','System' classes while also making the data uniform across 
#given 4 classes.

parser = English()
punctuations = string.punctuation

post = nps_chat.xml_posts()
post_greet = [p for p in post if p.get('class') == 'Greet']
post_system = [p for p in post if p.get('class') == 'System']
post_statement = [p for p in post if p.get('class') == 'Statement']
post_emotion = [p for p in post if p.get('class') == 'Emotion']

train_posts = [p.text.strip().lower() for p in post_greet[:900]] +\
              [p.text.strip().lower() for p in post_system[:1000]] +\
              [p.text.strip().lower() for p in post_statement[:1000]] +\
              [p.text.strip().lower() for p in post_emotion[:800]]

val_posts = [p.text.strip().lower() for p in post_greet[900:1100]] +\
            [p.text.strip().lower() for p in post_system[1000:1200]] +\
            [p.text.strip().lower() for p in post_statement[1000:1200]] +\
            [p.text.strip().lower() for p in post_emotion[800:1000]]

test_posts = [p.text.strip().lower() for p in post_greet[1100:]] +\
             [p.text.strip().lower() for p in post_system[1200:1332]] +\
             [p.text.strip().lower() for p in post_statement[1200:1385]] +\
             [p.text.strip().lower() for p in post_emotion[1000:]]

y_train = [p.get('class') for p in post_greet[:900]] +\
          [p.get('class') for p in post_system[:1000]] +\
          [p.get('class') for p in post_statement[:1000]] +\
          [p.get('class') for p in post_emotion[:800]]

y_val = [p.get('class') for p in post_greet[900:1100]] +\
        [p.get('class') for p in post_system[1000:1200]] +\
        [p.get('class') for p in post_statement[1000:1200]] +\
        [p.get('class') for p in post_emotion[800:1000]]

y_test = [p.get('class') for p in post_greet[1100:]] +\
         [p.get('class') for p in post_system[1200:1332]] +\
         [p.get('class') for p in post_statement[1200:1385]] +\
         [p.get('class') for p in post_emotion[1000:]]

train_p = [(post,class_) for post, class_ in zip(train_posts,y_train)]
val_p = [(post,class_) for post, class_ in zip(val_posts,y_val)]
test_p = [(post,class_) for post, class_ in zip(test_posts,y_test)]

shuffle(train_p)
shuffle(val_p)
shuffle(test_p)

train_posts = [x[0] for x in train_p]
val_posts = [x[0] for x in val_p]
test_posts = [x[0] for x in test_p]

y_train = [x[1] for x in train_p]
y_val = [x[1] for x in val_p]
y_test = [x[1] for x in test_p]

labels = ['Greet','Emotion','Statement','System']

#one hot encoding for labels
def one_hot(labels,tag_list):
  coding = {tag:i for tag,i in zip(tag_list,range(len(tag_list)))}
  one_hot = []
  for tag in labels:
    temp = [0 for i in tag_list]
    temp[coding[tag]] = 1
    one_hot.append(temp)
  return one_hot

#for tokenizing texts
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    #tokens = [tok for tok in tokens if (tok not in stopwords)]    
    return tokens

#Using tfidf vectorizer to vectorize text
tfidf = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,2))

data = tfidf.fit_transform(train_posts + val_posts + test_posts).todense()

train = data[:3700]
val = data[3700:4500]
test = data[4500:]

data_labels = one_hot(y_train + y_val + y_test, labels)
train_labels = data_labels[:3700]
val_labels = data_labels[3700:4500]
test_labels = data_labels[4500:]

#defining tensorflow graphs. I am using a 4 layered feedforward network.
input_dim = train.shape[1]
hidden_dim = 1024
output_dim = len(labels)
lr = 0.001
n_epoche = 1000
batch_size = 500
alpha = 0

#Loading inputs and outputs
x = tf.placeholder(tf.float32, [None, input_dim])
y = tf.placeholder(tf.float32, [None, output_dim])

#Weights and biases
W = {'h1':tf.Variable(tf.random_normal([input_dim,hidden_dim])),
     'h2':tf.Variable(tf.random_normal([hidden_dim,hidden_dim])),
     'h3':tf.Variable(tf.random_normal([hidden_dim,hidden_dim])),
     'h4':tf.Variable(tf.random_normal([hidden_dim,hidden_dim])),
     'out':tf.Variable(tf.random_normal([hidden_dim,output_dim]))
    }

b = {'b1':tf.Variable(tf.random_normal([hidden_dim])),
     'b2':tf.Variable(tf.random_normal([hidden_dim])),
     'b3':tf.Variable(tf.random_normal([hidden_dim])),
     'b4':tf.Variable(tf.random_normal([hidden_dim])),
     'b_out':tf.Variable(tf.random_normal([output_dim]))
    }

#forward pass
z2 = tf.add(tf.matmul(x,W['h1']),b['b1'])
a2 = tf.nn.tanh(z2)
z3 = tf.add(tf.matmul(a2,W['h2']),b['b2'])
a3 = tf.nn.tanh(z3)
z4 = tf.add(tf.matmul(a3,W['h3']),b['b3'])
a4 = tf.nn.tanh(z4)
z5 = tf.add(tf.matmul(a4,W['h4']),b['b4'])
a5 = tf.nn.tanh(z5)
z6 = tf.add(tf.matmul(a5,W['out']),b['b_out'])
y_pred = tf.nn.tanh(z6)


#cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,\
                                                              logits = y_pred))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)

#prediction and acc
pred = tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(pred,tf.float32))

#initializing
hist = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
init = tf.initialize_all_variables()

#Running the graph
print("Starting computation....")
with tf.Session() as sess:
    sess.run(init)
    for epoches in range(n_epoche):
      	for epoche in range(train.shape[0]//batch_size):
	        offset = (epoche * batch_size) % (len(train_labels) - batch_size)
	        batch_x = train[offset:(offset+batch_size)]
	        batch_y = train_labels[offset:(offset+batch_size)]
	        
	        o,c = sess.run([optimizer,cost],\
	                                  feed_dict = {x:batch_x, y:batch_y})
        
      	train_acc = accuracy.eval({x:train, y:train_labels})
      	if epoches % 100 == 0 or epoches == 0:
        	val_loss, val_pred = sess.run([cost,pred],\
                                      feed_dict = {x:val,y:val_labels})
        	val_acc = accuracy.eval({x:val,y:val_labels})
        	message = "epoche {:0d} : train_loss = {:02.02f}, train_acc = {:02.4f}, val_loss = {:02.02f}, val_acc = {:02.04f}".format(epoches, c, train_acc, val_loss, val_acc)
        	test_acc = accuracy.eval({x:test, y:test_labels})
        	print(message)
      	hist['train_acc'].append(train_acc)
      	hist['train_loss'].append(c)
      	hist['val_acc'].append(val_acc)
      	hist['val_loss'].append(val_loss)
      	if epoches == n_epoche-1: 
        	print("Training done!")
        	print("Test_Accuracy:",test_acc)

plt.figure(1)
plt.plot(hist['train_acc'])
plt.plot(hist['val_acc'])
plt.show()

plt.figure(2)
plt.plot(hist['train_loss'])
plt.plot(hist['val_loss'])
plt.show()
#The above code produces the following output:-
#Starting computation....
#epoche 0 : train_loss = 1.54, train_acc = 0.2992, val_loss = 1.50, val_acc = 0.3350
#epoche 100 : train_loss = 0.72, train_acc = 0.7805, val_loss = 1.05, val_acc = 0.6187
#epoche 200 : train_loss = 0.62, train_acc = 0.8105, val_loss = 1.01, val_acc = 0.6350
#epoche 300 : train_loss = 0.54, train_acc = 0.8654, val_loss = 0.94, val_acc = 0.6538
#epoche 400 : train_loss = 0.51, train_acc = 0.8716, val_loss = 0.94, val_acc = 0.6812
#epoche 500 : train_loss = 0.48, train_acc = 0.8886, val_loss = 0.93, val_acc = 0.6712
#epoche 600 : train_loss = 0.46, train_acc = 0.9016, val_loss = 0.90, val_acc = 0.7075
#epoche 700 : train_loss = 0.47, train_acc = 0.9151, val_loss = 0.91, val_acc = 0.6938
#epoche 800 : train_loss = 0.48, train_acc = 0.9135, val_loss = 0.89, val_acc = 0.7025
#epoche 900 : train_loss = 0.45, train_acc = 0.9243, val_loss = 0.88, val_acc = 0.7113
#Training done!
#Test_Accuracy: 0.7696793