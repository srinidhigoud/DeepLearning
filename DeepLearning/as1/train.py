

'''
Deep Learning Programming Assignment 1
--------------------------------------
Name:
Roll No.:

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
import numpy as np

batch_size = 100
epoch = 15
input_size = 784
hidden_size =100
output_size = 10

def load_batch(batch_num,trainX,trainY):
    X_1,Y_1 = trainX[(batch_num)*batch_size:(batch_num+1)*batch_size],trainY[(batch_num)*batch_size:(batch_num+1)*batch_size]
    return X_1.reshape(batch_size,input_size),Y_1

# def accuracy(w_ih,w_ho,b_ih,b_ho,X,Y):
#     #X.reshape([X.shape[0],input_size])
#     X = np.reshape(X,[-1,input_size])
#     print X.shape
#     hidden_layer_output = np.dot(X,w_ih) + b_ih #input to hidden
#     sigmoid_output = 1./(1 + np.exp(-hidden_layer_output)) #sigmoid layer
#     output_layer_output = np.dot(sigmoid_output,w_ho) + b_ho #hidden to output layer
#     softmax_output = np.exp(output_layer_output)/np.sum(np.exp(output_layer_output),axis=1,keepdims = True)
#     labels = np.argmax(softmax_output,axis=1)
#     print  np.mean((labels==Y))*100.0

def train(trainX, trainY):
    W_ih = -0.01*np.random.randn(input_size,hidden_size)
    b_ih = np.zeros((1,hidden_size))
    W_ho = -0.01*np.random.randn(hidden_size,output_size)
    b_ho = np.zeros((1,output_size))
    learning_rate = 0.002
    for i in range(epoch):
        if(i>5):
            learning_rate = 0.001
        if(i>10):
            learning_rate = 0.0005
        for j in range(trainX.shape[0]/batch_size):
            X,Y = load_batch(j,trainX,trainY) #load in batches of 100
            hidden_layer_output = np.dot(X,W_ih) + b_ih #input to hidden
            #activation_output = 1./(1 + np.exp(-hidden_layer_output)) #sigmoid activation
            activation_output = np.maximum(0,hidden_layer_output) #ReLU activation
            #activation_output = np.tanh(hidden_layer_output) #tanh activation
            output_layer_output = np.dot(activation_output,W_ho) + b_ho #hidden to output layer
            softmax_output = np.exp(output_layer_output)/np.sum(np.exp(output_layer_output),axis=1,keepdims = True)
            log_probs = -np.log(softmax_output[np.arange(batch_size),Y])
            loss = np.sum(log_probs)/batch_size
            dsoftmax = softmax_output
            dsoftmax[np.arange(batch_size),Y] -= 1
            dsoftmax /= batch_size
            dw_ho = np.dot(activation_output.T,dsoftmax)
            db_ho = np.sum(dsoftmax,axis=0,keepdims=True)
            dhidden = np.dot(dsoftmax,W_ho.T)
            #dactivation = activation_output*(1-activation_output)*dhidden #sigmoid backprop
            dhidden[activation_output<=0] = 0 #ReLU backprop
            dactivation = dhidden
            #dactivation = (1 - np.square(activation_output))*dhidden #tanh backprop
            dw_ih = np.dot(X.T,dactivation)
            db_ih = np.sum(dactivation,axis=0,keepdims=True)
            #params update   
            W_ho += -learning_rate*dw_ho
            b_ho += -learning_rate*db_ho
            W_ih += -learning_rate*dw_ih
            b_ih += -learning_rate*db_ih

        print 'epoch ',i+1,' loss = ',loss
    np.savetxt('weights/W_ih.txt',W_ih)
    np.savetxt('weights/W_ho.txt',W_ho)
    np.savetxt('weights/b_ih.txt',b_ih)
    np.savetxt('weights/b_ho.txt',b_ho)
    # accuracy(W_ih,W_ho,b_ih,b_ho,trainX,trainY)

                


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
    W_ih = np.genfromtxt('weights/W_ih.txt')
    W_ho = np.genfromtxt('weights/W_ho.txt')
    b_ih = np.genfromtxt('weights/b_ih.txt')
    b_ho = np.genfromtxt('weights/b_ho.txt')
    X = np.reshape(testX,[-1,input_size])
    hidden_layer_output = np.dot(X,W_ih) + b_ih #input to hidden
    #activation_output = np.tanh(hidden_layer_output) #Tanh activation
    activation_output = np.maximum(0,hidden_layer_output) #ReLU activation
    #activation_output = 1./(1 + np.exp(-hidden_layer_output)) #sigmoid activation
    output_layer_output = np.dot(activation_output,W_ho) + b_ho #hidden to output layer
    softmax_output = np.exp(output_layer_output)/np.sum(np.exp(output_layer_output),axis=1,keepdims = True)
    labels = np.argmax(softmax_output,axis=1)
    return labels
