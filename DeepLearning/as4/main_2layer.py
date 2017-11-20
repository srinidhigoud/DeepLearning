import os
import tensorflow as tf 
import math
import numpy as np
import ptb_reader
import pickle

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("test","../data/ptb.test.txt", "Path to test file")

FLAGS = flags.FLAGS

def init_weight(dim_in, dim_out, name=None, stddev=1.0):
	return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

def init_bias(dim_out, name=None):	
	return tf.Variable(tf.zeros([dim_out]), name=name)

data_dir = '../data'

with open('weights/word_to_id.pkl','rb') as f:
	word_to_id = pickle.load(f)

with open('weights/vocab_size.pkl','rb') as f:
	vocab_size = pickle.load(f)

test_data = ptb_reader._file_to_word_ids(FLAGS.test, word_to_id)

batch_size = 20
lstm_steps = 40
embedding_dim = 400
lstm_dim = 400
epochs = 4
learning_rate = 0.001

with tf.variable_scope(tf.get_variable_scope()) as scope:
	W_embedding = tf.Variable(tf.random_uniform([vocab_size,embedding_dim], -1.0, 1.0), name='Wemb')

	init_Wh1 = init_weight(lstm_dim,4*lstm_dim)
	init_Wx1 = init_weight(embedding_dim,4*lstm_dim)
	init_b = init_bias(4*lstm_dim)
	init_Wh2 = init_weight(lstm_dim,4*lstm_dim)
	init_Wx2 = init_weight(lstm_dim,4*lstm_dim)

	decode_lstm_W = init_weight(lstm_dim,lstm_dim)
	decode_lstm_b = init_bias(lstm_dim)

	decode_word_W = init_weight(lstm_dim,vocab_size)
	decode_word_b = init_bias(vocab_size)

	inputs = tf.placeholder("int32", [batch_size,lstm_steps])
	targets = tf.placeholder("int32", [batch_size,lstm_steps])

	h1 = tf.zeros([batch_size,lstm_dim])
	c1 = tf.zeros([batch_size,lstm_dim])
	h2 = tf.zeros([batch_size,lstm_dim])
	c2 = tf.zeros([batch_size,lstm_dim])

	loss = 0.0

	for step in range(lstm_steps):
		if step!=0:
			tf.get_variable_scope().reuse_variables()
		word_emb = tf.nn.embedding_lookup(W_embedding,inputs[:,step])

		labels = tf.expand_dims(targets[:,step], 1)
		indices = tf.expand_dims(tf.range(0,batch_size,1),1)
		concated = tf.concat([indices, labels],1)
		onehot_labels = tf.sparse_to_dense(concated,[batch_size,vocab_size],1.0,0.0)

		hidden_output = tf.matmul(h1,init_Wh1)+tf.matmul(word_emb,init_Wx1)
		i1,f1,o1,new_c1 = tf.split(hidden_output,4,1)
		i1 = tf.nn.sigmoid(i1)
		f1 = tf.nn.sigmoid(f1)
		o1 = tf.nn.sigmoid(o1)
		new_c1 = tf.nn.tanh(new_c1)
		c1 = f1*c1 + i1*new_c1
		h1 = o1*tf.nn.tanh(new_c1)

		lstm_output = tf.matmul(h2,init_Wh2)+tf.matmul(h1,init_Wx2)
		i2,f2,o2,new_c2 = tf.split(lstm_output,4,1)
		i2 = tf.nn.sigmoid(i2)
		f2 = tf.nn.sigmoid(f2)
		o2 = tf.nn.sigmoid(o2)
		new_c2 = tf.nn.tanh(new_c2)
		c2 = f2*c2 + i2*new_c2
		h2 = o2*tf.nn.tanh(new_c2)


		# logits = tf.matmul(h2,decode_lstm_W)+decode_lstm_b
		# logits = tf.nn.relu(logits)
		# logits = tf.nn.dropout(logits,0.5)

		logit_words = tf.matmul(h2,decode_word_W) + decode_word_b
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words,labels=onehot_labels)
		current_loss = tf.reduce_sum(cross_entropy)
		loss = loss + current_loss
	loss = tf.div(loss,batch_size)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()
sess = tf.InteractiveSession()

def test(data):

	saver = tf.train.import_meta_graph('weights/ptb_lstm_model.meta')
	saver.restore(sess, 'weights/ptb_lstm_model')
	epoch_size = ((len(data) // batch_size) - 1) // lstm_steps

	loss1 = 0.0
	iters = 0
	# tf.global_variables_initializer().run()
	for step,(x,y) in enumerate(ptb_reader.ptb_iterator(data,batch_size,lstm_steps)):
		# print step
		loss_temp = sess.run(loss,feed_dict={inputs:x,targets:y})
		loss1 += loss_temp
		iters += lstm_steps
		perplexity = np.exp(loss1/iters)

	return perplexity


def train(data):

	# saver = tf.train.import_meta_graph('weights/ptb_lstm_model.meta')
	# saver.restore(sess, 'weights/ptb_lstm_model')

	epoch_size = ((len(data) // batch_size) - 1) // lstm_steps 

	loss1 = 0.0
	iters = 0
	test_perplexity_prev = 10000000.0
	tf.global_variables_initializer().run()
	for ep in range(epochs):
		for step,(x,y) in enumerate(ptb_reader.ptb_iterator(data,batch_size,lstm_steps)):
			loss_temp,_ = sess.run([loss,train_step],feed_dict={inputs:x,targets:y})
			loss1 += loss_temp
			iters += lstm_steps
			perplexity = np.exp(loss1/iters)

			# if step%10==0:
			progress = (step/float(epoch_size))*100.0
			print("%d %.1f%% Perplexity: %.3f (Loss: %.3f)" % (ep,progress, perplexity,loss1/iters))
		# saver.save(sess,'weights/ptb_lstm_model')
		# print 'Trained model saved'
		# test_perplexity = test(test_data)
		# print("Test Perplexity: %.3f" % test_perplexity)
		loss_test = 0.0
		iters_test = 0
		for step,(x,y) in enumerate(ptb_reader.ptb_iterator(test_data,batch_size,lstm_steps)):
			print step
			loss_temp = sess.run(loss,feed_dict={inputs:x,targets:y})
			loss_test += loss_temp
			iters_test += lstm_steps
		test_perplexity = np.exp(loss_test/iters_test)
		if test_perplexity>test_perplexity_prev:
			break
		print("Test Perplexity: %.3f" % test_perplexity)
		test_perplexity_prev = test_perplexity

	saver.save(sess,'weights/ptb_lstm_model')
	print 'Trained model saved'
	return perplexity,loss1/iters




# train_preplexity,train_loss = train(train_data)
# print("Train Perplexity: %.3f" % train_preplexity)
# test_perplexity = test(test_data)
# print("Test Perplexity: %.3f" % test_perplexity)
# valid_perplexity = test(valid_data)
# print("Validation Perplexity: %.3f" % valid_perplexity)
def main(argv):
	print(test(test_data))

if __name__ == "__main__":
	tf.app.run()





 




















