import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
import xml_reader


# Global Variable
TRAINING_INDEX = 1315
CATEGORY_INDEX = {
    '': 0,
    'RESTAURANT#GENERAL': 1,
    'SERVICE#GENERAL': 2,
    'FOOD#QUALITY': 3,
    'DRINKS#QUALITY': 4,
    'DRINKS#STYLE_OPTIONS': 5,
    'FOOD#PRICES': 6, 'DRINKS#PRICES': 7,
    'RESTAURANT#MISCELLANEOUS': 8,
    'FOOD#STYLE_OPTIONS': 9,
    'LOCATION#GENERAL': 10,
    'RESTAURANT#PRICES': 11,
    'AMBIENCE#GENERAL': 12
}

TRAIN_DATA = {
    '': [],
    'RESTAURANT#GENERAL': [],
    'SERVICE#GENERAL': [],
    'FOOD#QUALITY': [],
    'DRINKS#QUALITY': [],
    'DRINKS#STYLE_OPTIONS': [],
    'FOOD#PRICES': [],
    'DRINKS#PRICES': [],
    'RESTAURANT#MISCELLANEOUS': [],
    'FOOD#STYLE_OPTIONS': [],
    'LOCATION#GENERAL': [],
    'RESTAURANT#PRICES': [],
    'AMBIENCE#GENERAL': []
}

TEST_DATA = {
    '': [],
    'RESTAURANT#GENERAL': [],
    'SERVICE#GENERAL': [],
    'FOOD#QUALITY': [],
    'DRINKS#QUALITY': [],
    'DRINKS#STYLE_OPTIONS': [],
    'FOOD#PRICES': [],
    'DRINKS#PRICES': [],
    'RESTAURANT#MISCELLANEOUS': [],
    'FOOD#STYLE_OPTIONS': [],
    'LOCATION#GENERAL': [],
    'RESTAURANT#PRICES': [],
    'AMBIENCE#GENERAL': []
}

# Word Embedding
sem_eval_restaurant_model = Word2Vec.load("sem_eval_restaurant_model.vec")

# Parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 1
n_epochs = 15


max_sentence_len = 74
h_filter = 5
n_filter = 50
embed_vec_len = 100
n_hidden = 30  # for hidden dense layer
n_classes = 2


# Data
text_data_lst, category_data_lst = xml_reader.load_data_for_task_1_2()

for text_data, category_data in zip(text_data_lst[:TRAINING_INDEX], category_data_lst[:TRAINING_INDEX]):
    embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in text_data]

    for _ in range(max_sentence_len - len(embedded_sentence_data)):
        embedded_sentence_data.append([0.0] * 100)

    if len(category_data) == 0:
        TRAIN_DATA[""].append(embedded_sentence_data)
    else:
        for category in category_data:
            TRAIN_DATA[category].append(embedded_sentence_data)

for text_data, category_data in zip(text_data_lst[:TRAINING_INDEX], category_data_lst[:TRAINING_INDEX]):
    embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in text_data]

    for _ in range(max_sentence_len - len(embedded_sentence_data)):
        embedded_sentence_data.append([0.0] * 100)

    if len(category_data) == 0:
        TEST_DATA[""].append(embedded_sentence_data)
    else:
        for category in category_data:
            TEST_DATA[category].append(embedded_sentence_data)


# Model
class SemEvalSlot1CNNModel(object):
    def __init__(self, label):
        self.label = label

        # Session
        self.sess = tf.Session()

        # Data
        self.input_data = []
        self.output_data = []

        # Init data set
        for data_label, train_data in TRAIN_DATA.items():
            for data in train_data:
                if label == data_label:
                    self.output_data.append([1, 0])
                else:
                    self.output_data.append([0, 1])
                self.input_data.append(data)

        # Input place holder
        self.X = tf.placeholder(tf.float32, [None, max_sentence_len, embed_vec_len, 1])
        self.Y = tf.placeholder(tf.float32, [None, 2])

        # Layer 1 - CNN
        self.W1 = tf.Variable(tf.random_normal([h_filter, embed_vec_len, 1, n_filter], stddev=0.01))

        self.L1 = tf.nn.conv2d(self.X, self.W1,
                               strides=[1, 1, 1, 1],
                               padding="VALID")
        self.L1 = tf.nn.tanh(self.L1)
        self.L1 = tf.nn.max_pool(self.L1,
                                 ksize=[1, max_sentence_len - h_filter + 1, 1, 1],
                                 strides=[1, 1, 1, 1],
                                 padding="VALID")

        # Layer 2 - Hidden Layer
        self.W2 = tf.Variable(tf.random_normal([n_filter, n_hidden]))

        self.L2 = tf.reshape(self.L1, [-1, n_filter])
        self.L2 = tf.matmul(self.L2, self.W2)
        self.L2 = tf.nn.relu(self.L2)

        # Layer 3 - FC
        self.W3 = tf.Variable(tf.random_normal([n_hidden, n_classes]))
        self.model = tf.matmul(self.L2, self.W3)

    def train(self):
        # Training
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Global Variable Initialize
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for batch_x, batch_y in zip(self.input_data, self.output_data):
                batch_x = np.expand_dims(batch_x, -1)
                self.sess.run([optimizer, cost], feed_dict={self.X: [batch_x], self.Y: [batch_y]})
            print("Label : {}, Epoch: {}".format(self.label, epoch + 1))

    def predict(self, text_lst):
        text_lst = np.expand_dims(text_lst, -1)
        d = self.sess.run(self.model, feed_dict={self.X: [text_lst]})
        print(self.sess.run(tf.one_hot(tf.argmax(d, 1), 2)))


if __name__ == "__main__":
    model0 = SemEvalSlot1CNNModel('')
    model0.train()
    model0.predict(TEST_DATA[''][0])
    model0.predict(TEST_DATA[''][1])
    model0.predict(TEST_DATA[''][2])
    model0.predict(TEST_DATA[''][3])
    model0.predict(TEST_DATA[''][4])
    model0.predict(TEST_DATA['RESTAURANT#GENERAL'][0])
    model0.predict(TEST_DATA['RESTAURANT#GENERAL'][1])
    model0.predict(TEST_DATA['RESTAURANT#GENERAL'][2])
    model0.predict(TEST_DATA['RESTAURANT#GENERAL'][3])
    model0.predict(TEST_DATA['RESTAURANT#GENERAL'][4])
    model0.predict(TEST_DATA['SERVICE#GENERAL'][0])
    model0.predict(TEST_DATA['SERVICE#GENERAL'][1])
    model0.predict(TEST_DATA['SERVICE#GENERAL'][2])
    model0.predict(TEST_DATA['SERVICE#GENERAL'][3])
    model0.predict(TEST_DATA['SERVICE#GENERAL'][4])

