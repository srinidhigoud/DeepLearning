import tensorflow as tf 
import numpy as np 
import random
import pickle
import math
import csv


def train_analogy(dataset,analogy_dataset):
	anaSoln = "Q1/analogySolution.csv"
	# print("started")

	# with open('dataset_pos_neg.pkl','rb') as f:
		# dataset = pickle.load(f)

	# print("Successfully loaded the dataset")
	total_size = len(dataset)
	size = len(dataset)/5

	trainset = []
	testset = []

	trainset.append(dataset[:4*size])
	trainset.append(dataset[size:])
	trainset.append(dataset[2*size:]+dataset[:size])
	trainset.append(dataset[3*size:]+dataset[:2*size])
	trainset.append(dataset[4*size:]+dataset[:3*size])

	testset.append(dataset[4*size:])
	testset.append(dataset[:size])
	testset.append(dataset[size:2*size])
	testset.append(dataset[2*size:3*size])
	testset.append(dataset[3*size:4*size])



	 
	# print("Dataset divided in 5 parts")

	batch_size = 100


	def model(x):
		W1 = tf.Variable(tf.random_normal((600,100), stddev=0.1))
		b1 = tf.Variable(tf.zeros(100))

		W2 = tf.Variable(tf.random_normal((100,30), stddev=0.1))
		b2 = tf.Variable(tf.zeros(30))

		W3 = tf.Variable(tf.random_normal((30,1), stddev=0.1))
		b3 = tf.Variable(tf.zeros(1))

		h1 = tf.nn.relu(tf.matmul(x,W1)+b1)
		h2 = tf.nn.relu(tf.matmul(h1,W2)+b2)
		score  = tf.matmul(h2,W3)
		return score

	x_pos = tf.placeholder(tf.float32,[None,600])
	x_neg = tf.placeholder(tf.float32,[None,600])
	score_pos = model(x_pos)
	score_neg = model(x_neg)

	loss = tf.reduce_mean(tf.maximum(0.0,1.0-score_pos+score_neg))

	train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	def load_batch(dataset,num,batch_size):
		pos = np.zeros((batch_size,600))
		neg = np.zeros((batch_size,600))
		for k in range(batch_size):
			if(num < len(dataset)):
				pos[k] = dataset[num]['pos']['vector']
				neg[k] = dataset[num]['neg']['vector']
				num += 1
			else:
				num = 0
				k = k-1
		return pos,neg,num

	def accuracy(testset,sess):
		pos = np.zeros((len(testset),600))
		neg = np.zeros((len(testset),600))
		print "yusss"
		for k in range(len(testset)):
			pos[k] = testset[k]['pos']['vector']
			neg[k] = testset[k]['neg']['vector']
		num_pos,num_neg = sess.run([score_pos,score_neg],feed_dict={x_pos:pos,x_neg:neg})
		print num_pos[:10],num_neg[:10]
		accuracy = np.mean(num_pos>num_neg)*100
		return accuracy

	for i in range(5):
		batch_num = 0
		batch_pos1,batch_neg1,n = load_batch(trainset[i],0,len(trainset[i]))
		for j in range(100):
			batch_pos,batch_neg,batch_num = load_batch(trainset[i],batch_num,batch_size)
			sess.run(train_step,feed_dict={x_pos:batch_pos,x_neg:batch_neg})
			# if j%10==0:
				
				# print(sess.run(loss,feed_dict={x_pos:batch_pos1,x_neg:batch_neg1}))
		# print len(testset[i])
		accuracy1 = accuracy(testset[i],sess)
		# print 'accuracy=',accuracy1
		tf.global_variables_initializer().run()

	# with open('analogy_dataset_questions.pkl','rb') as f:
	 	# analogy_dataset = pickle.load(f)

	dataset_test = analogy_dataset['dataset']
	ground_truth_test = analogy_dataset['ground_truth']
	words = analogy_dataset['words']

	scores = []
	for i in range(dataset_test.shape[0]):
		scores.append(np.argmax(sess.run(score_pos,feed_dict={x_pos:dataset_test[i],x_neg:np.zeros((5,600))})))
	scores = np.array(scores)
	outputfile = open(anaSoln,'wb')
	writer_output = csv.writer(outputfile,delimiter=',')
	for i in range(len(scores)):
		writer_output.writerow([words[i][0],words[i][int(ground_truth_test[i])],words[i][scores[i]]])

	return accuracy1,np.mean(scores==ground_truth_test)*100





	




















