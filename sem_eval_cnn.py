import tensorflow as tf
import xml_reader
import numpy as np
#from gensim.models import Word2Vec
#from gensim.models.wrappers.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
import os

TRAINING_INDEX = 1315

text_list, category = xml_reader.load_data_for_task_1_2('data/ABSA16_Restaurants_Train_SB1_v2.xml')
#eng_model = KeyedVectors.load_word2vec_format('wiki.simple.vec');
eng_model = KeyedVectors.load_word2vec_format('wiki.en.vec')
print('load complete!')
index_dict={
''				: 0, 
'RESTAURANT#GENERAL'		: 1, 
'SERVICE#GENERAL'		: 2,
'FOOD#QUALITY'			: 3,
'DRINKS#QUALITY'		: 4,
'DRINKS#STYLE_OPTIONS'		: 5,
'FOOD#PRICES'			: 6,
'DRINKS#PRICES'			: 7,
'RESTAURANT#MISCELLANEOUS'	: 8,
'FOOD#STYLE_OPTIONS'		: 9,
'LOCATION#GENERAL'		: 10,
'RESTAURANT#PRICES'		: 11,
'AMBIENCE#GENERAL'		: 12
}

m = 5 # Filter size
d = eng_model.vector_size # Word Embedding Dimension
sent_len = max(map(len, text_list)); # Number of Words in the Longest Sentence
n = 300 # Number of Filters
h = 100 # Number of Hidden Units

#text_list = [[word for word in text if word != ';' and word != '3'] for text in text_list]
text_embeds=[]
for text in text_list :
	embeds_i=[]
	for word in text :
		try:
			embeds_i.append(eng_model.wv[word.lower()])
		except KeyError:
			embeds_i.append([0.0]*d)
	for i in range(sent_len - len(text)) :
		embeds_i.append([0.0]*d)

	text_embeds.append(embeds_i)

#text_embeds = [[try: eng_model.wv[word.lower() except KeyError: [0.0] * 300] for word in text] for text in text_list]
#text_embeds_pad = [[embed + [[0.0] * d] * (sent_len - len(embed))] for embed in text_embeds]

category_prob = []
for i in category :
	prob_i = [0.0] * 13
	category_num = [0] if i == [] else [index_dict[j] for j in i]
	for j in category_num :
		prob_i[j] = 1. / len(category_num)
	category_prob.append(prob_i)


# Layer Define
X=tf.placeholder(tf.float32, [None, sent_len, d, 1])
Y=tf.placeholder(tf.float32, [None, 13])

W1=tf.Variable(tf.random_normal([m, d, 1, n], stddev=0.01))
L1=tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID')
L1=tf.nn.tanh(L1)
L1=tf.nn.max_pool(L1, ksize=[1, sent_len - m + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

W2=tf.Variable(tf.random_normal([n, h], stddev=0.01))
L2=tf.reshape(L1, [-1, n])
L2=tf.matmul(L2, W2);
L2=tf.nn.relu(L2)

W3=tf.Variable(tf.random_normal([h, 13], stddev=0.01))
model=tf.matmul(L2, W3);

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y)) + 0.01 * (tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3));
optimizer=tf.train.AdadeltaOptimizer(rho=0.95, epsilon=1e-06).minimize(cost)

#Train
init = tf.global_variables_initializer()
sess = tf.Session();
sess.run(init);

batch_size=50
total_batch=int(TRAINING_INDEX / batch_size) + 1;

for epoch in range(15):
	for i in range(total_batch):
		start = epoch * batch_size
		end = min(TRAINING_INDEX, (epoch+1)*batch_size)
		batch_xs = text_embeds[start:end]
		batch_xs = np.expand_dims(np.array(batch_xs), -1)
		#batch_xs = tf.expand_dims(batch_xs, -1)
		batch_ys = category_prob[start:end]
		sess.run([optimizer, cost], feed_dict={X:batch_xs, Y: batch_ys})

# Result
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
test_text = np.expand_dims(np.array(text_embeds[TRAINING_INDEX:]), -1)
print(sess.run(accuracy, feed_dict={X:test_text, Y:category_prob[TRAINING_INDEX:]}))
