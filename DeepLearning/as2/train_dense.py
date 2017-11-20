'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: SRINIDHI GOUD 
Roll No.: 13EC10042

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import tensorflow as tf 

batch_size = 100
epoch = 20
input_size = 784
hidden_size =100
output_size = 10


x = tf.placeholder(tf.float32,[None,input_size])
W1 = tf.Variable(tf.truncated_normal([input_size,hidden_size],stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_size]))
W2 = tf.Variable(tf.truncated_normal([hidden_size,output_size],stddev=0.1))
b2 = tf.Variable(tf.zeros([output_size]))
h = tf.nn.relu(tf.matmul(x,W1)+b1)
y = tf.matmul(h,W2)+b2
y_ = tf.placeholder(tf.float32,[None,output_size])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver = tf.train.Saver()
sess = tf.InteractiveSession()

def onehot_encoding(Y_batch):
	Y_ = np.zeros((Y_batch.shape[0],10),dtype=float)
	for i in range(Y_batch.shape[0]):
		Y_[i][Y_batch[i]] = 1
	return Y_

def load_batch(trainX,trainY,batch_num):
	X1,Y1 = trainX[batch_num*batch_size:(batch_num+1)*batch_size],trainY[batch_num*batch_size:(batch_num+1)*batch_size]
	X = X1.reshape(batch_size,input_size)
	Y = onehot_encoding(Y1)
	return X,Y

def train(trainX, trainY):
    '''
    Complete this function.
    '''
    tf.global_variables_initializer().run()
    # print sess.run(W2)
    for i in range(epoch):
    	for j in range(trainX.shape[0]/batch_size):
    	   batchX,batchY = load_batch(trainX,trainY,j)
    	   sess.run(train_step,feed_dict={x:batchX ,y_:batchY})
    	print sess.run(cross_entropy,feed_dict={x:trainX.reshape(trainX.shape[0],input_size),y_:onehot_encoding(trainY)})

    saver.save(sess,'DL_Assignment3_weights/modeldense')
    print 'model saved'



def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''
    saver = tf.train.import_meta_graph('DL_Assignment3_weights/modeldense.meta')
    saver.restore(sess, 'DL_Assignment3_weights/modeldense')
    output = sess.run(y,feed_dict={x:testX.reshape(testX.shape[0],784)})

    return np.argmax(output,1)
