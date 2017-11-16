import tensorflow as tf 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.model_selection import train_test_split as tts

enc = OHE()

iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape((150,1))
enc.fit(y)
y = enc.transform(y).toarray()

X,X_test,y,y_test = tts(X,y,test_size = 0.33, random_state = 39)

#Initializing 
tf.reset_default_graph()


#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[None,4],dtype=tf.float32)
y_ = tf.placeholder(tf.float32, [None, 3])

W1 = tf.Variable(tf.random_uniform([4,10],minval = 0, maxval = 0.01))
b1 = tf.Variable(tf.random_uniform([1,10],minval = 0, maxval = 0.01))

hidden1 = tf.matmul(inputs1,W1)+b1

W2 = tf.Variable(tf.random_uniform([10,3],minval = 0,maxval = 0.01))
b2 =  tf.Variable(tf.random_uniform([1,3],minval = 0, maxval = 0.01))

out = tf.matmul(hidden1,W2)+b2
predict = tf.nn.softmax(out)

loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = predict)
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(len(X)):
	sess.run(updateModel,feed_dict = {inputs1: X[i].reshape((1,4)), y_: y[i].reshape((1,3))})

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={inputs1: X_test.reshape((-1,4)), y_: y_test.reshape((-1,3))}))