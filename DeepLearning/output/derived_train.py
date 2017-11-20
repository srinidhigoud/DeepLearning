import tensorflow as tf 
import numpy as np 
import random
import pickle
import math
from scipy import spatial

def derive_train(dataset,all_words)
# print("started")
# with open('derived_words_dict.pkl','rb') as f:
# 	dataset = pickle.load(f)

# with open('all_words_derived.pkl','rb') as f:
# 	all_words = pickle.load(f)

# print("Successfully loaded the dataset")

	def model(x):
		W1 = tf.Variable(tf.random_normal((200,256), stddev=0.1))
		b1 = tf.Variable(tf.zeros(256))

		W2 = tf.Variable(tf.random_normal((256,200), stddev=0.1))
		b2 = tf.Variable(tf.zeros(200))

		h1 = tf.nn.relu(tf.matmul(x,W1)+b1)
		output = tf.nn.relu(tf.matmul(h1,W2)+b2)
		return output

	x = tf.placeholder(tf.float32,[None,200])
	y = tf.placeholder(tf.float32,[None,200])
	predicted = model(x)
	loss = tf.reduce_mean(tf.square(tf.sub(y,predicted)))
	train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	cosine1 = []
	file = open("Q4/AnsModel.txt",'wb')
	for affix in dataset:
		x_train = dataset[affix]['fastText']['source']
		y_train = dataset[affix]['fastText']['derived']
		print affix
		for j in range(100):
			sess.run(train_step,feed_dict={x:x_train,y:y_train})
			if j%10 == 0:
				print(sess.run(loss,feed_dict={x:x_train,y:y_train}))
		output = sess.run(predicted,feed_dict={x:x_train,y:y_train})
		
		count=0
		for words in all_words[affix]:
			str1 = ' '.join(str(e) for e in output[count])
			count += 1
			print >> file,words+' '+str1
			# print words+' '+str1
		cosine = []
		for i in range(np.array(y_train).shape[0]):
			cosine.append(1-spatial.distance.cosine(output[i],y_train[i]))
		cosine1.append(np.mean(np.array(cosine)))
	cos1 = np.mean(cosine1)

	def model2(x):
		W1 = tf.Variable(tf.random_normal((350,40), stddev=0.1))
		b1 = tf.Variable(tf.zeros(40))

		W2 = tf.Variable(tf.random_normal((40,350), stddev=0.1))
		b2 = tf.Variable(tf.zeros(350))

		h1 = tf.nn.relu(tf.matmul(x,W1)+b1)
		output = tf.nn.relu(tf.matmul(h1,W2)+b2)
		return output

	x2 = tf.placeholder(tf.float32,[None,350])
	y2 = tf.placeholder(tf.float32,[None,350])
	predicted2 = model2(x2)
	loss2 = tf.reduce_mean(tf.square(tf.sub(y2,predicted2)))
	train_step2 = tf.train.AdamOptimizer(0.001).minimize(loss2)

	sess2 = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	cosine2 = []
	# file = open("Q4/AnsModel.txt",'wb')
	for affix in dataset:
		x_train = dataset[affix]['lazaridou']['source']
		y_train = dataset[affix]['lazaridou']['derived']
		print affix
		for j in range(100):
			sess2.run(train_step2,feed_dict={x2:x_train,y2:y_train})
			if j%10 == 0:
				print(sess2.run(loss2,feed_dict={x2:x_train,y2:y_train}))
		output = sess2.run(predicted2,feed_dict={x2:x_train,y2:y_train})
		
		# count=0
		# for words in all_words[affix]:
		# 	str1 = ' '.join(str(e) for e in output[count])
		# 	count += 1
		# 	print >> file,words+' '+str1
		# 	# print words+' '+str1
		cosine = []
		for i in range(np.array(y_train).shape[0]):
			if np.linalg.norm(output[i]) == 0:
				cosine.append(0)
			else:
				cosine.append(np.dot(output[i],y_train[i])/(np.linalg.norm(output[i])*np.linalg.norm(y_train[i])))
		cosine2.append(np.mean(np.array(cosine)))
	cos2 = np.mean(cosine2)
	return cos1,cos2
# return np.mean(cosine1)












