'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: 13EC10042
Roll No.: SRINIDHI GOUD

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np
import numpy as np
import tensorflow as tf 

batch_size = 100
epoch = 10

x = tf.placeholder(tf.float32,[None,28,28,1])
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32],stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
W1_fc = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b1_fc = tf.Variable(tf.zeros([1024]))
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h1_fc = tf.nn.relu(tf.matmul(h_pool2_flat,W1_fc)+b1_fc)
prob_dropout = tf.placeholder(tf.float32)
h1_fc_drop = tf.nn.dropout(h1_fc,prob_dropout)
W2_fc = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b2_fc = tf.Variable(tf.zeros([10]))
y = tf.matmul(h1_fc_drop,W2_fc)+b2_fc
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
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

def train(trainX, trainY,testX,testY):
    '''
    Complete this function.
    '''
    tf.global_variables_initializer().run()
    for i in range(epoch):

        # accuracy1 = 0
        # print('epoch comp epoch comp epoch comp epoch comp epoch comp epoch comp')
        # for k in range(testX.shape[0]/batch_size):
        #     batchtrX,batchtrY = testX[k*batch_size:(k+1)*batch_size],onehot_encoding(testY[k*batch_size:(k+1)*batch_size])
        #     accuracy1 = accuracy1 + sess.run(accuracy,feed_dict={x:batchtrX,y_:batchtrY,prob_dropout:1.0})
        #     print k
        # print('accuracy: ',accuracy1)

        for j in range(trainX.shape[0]/batch_size):
            batchX,batchY = trainX[j*batch_size:(j+1)*batch_size],onehot_encoding(trainY[j*batch_size:(j+1)*batch_size])
            sess.run(train_step,feed_dict={x:batchX,y_:batchY,prob_dropout:0.5})
            print i,j,sess.run(cross_entropy,feed_dict={x:batchX,y_:batchY,prob_dropout:1.0})




    saver.save(sess,'DL_Assignment3_weights/modelcnn_drop')
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
    saver = tf.train.import_meta_graph('DL_Assignment3_weights/modelcnn_drop.meta')
    saver.restore(sess, 'DL_Assignment3_weights/modelcnn_drop')
    output = np.zeros((10000,10))
    for i in range(100):
        output[i*100:(i+1)*100] = sess.run(y,feed_dict={x:testX[i*100:(i+1)*100],prob_dropout:1.0})
    return np.argmax(output,1)

