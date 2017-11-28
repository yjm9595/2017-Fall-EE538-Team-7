import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
import xml_reader


# Global Variable
TRAINING_INDEX = 1315
CATEGORY =['', 'RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#QUALITY', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
           'FOOD#PRICES', 'DRINKS#PRICES', 'RESTAURANT#MISCELLANEOUS', 'FOOD#STYLE_OPTIONS', 'LOCATION#GENERAL',
           'RESTAURANT#PRICES', 'AMBIENCE#GENERAL']

CATEGORY_INDEX = {
    '': 0,
    'RESTAURANT#GENERAL': 1,
    'SERVICE#GENERAL': 2,
    'FOOD#QUALITY': 3,
    'DRINKS#QUALITY': 4,
    'DRINKS#STYLE_OPTIONS': 5,
    'FOOD#PRICES': 6,
    'DRINKS#PRICES': 7,
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

# Word Embedding
sem_eval_restaurant_model = Word2Vec.load("sem_eval_restaurant_model.vec")

# Parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100
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

        total_batch = int(TRAINING_INDEX / batch_size) + 1

        for epoch in range(n_epochs):

            for i in range(total_batch):
                index_start = i * batch_size
                index_end = min(TRAINING_INDEX, (i + 1) * batch_size)
                batch_xs = self.input_data[index_start:index_end]
                batch_xs = np.expand_dims(batch_xs, -1)
                batch_ys = self.output_data[index_start:index_end]
                _, c = self.sess.run([optimizer, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys})
            print("Label : {}, Epoch: {}".format(self.label, epoch + 1))

    def predict(self, text_lst):
        text_lst = np.expand_dims(text_lst, -1)
        d = self.sess.run(self.model, feed_dict={self.X: [text_lst]})
        if self.sess.run(tf.argmax(d, 1))[0] == 0:
            return 1
        else:
            return 0


if __name__ == "__main__":
    models = [SemEvalSlot1CNNModel(cat) for cat in CATEGORY]

    # for model in models:
    #     model.train()
    model0 = SemEvalSlot1CNNModel("")
    model1 = SemEvalSlot1CNNModel('RESTAURANT#GENERAL')
    model2 = SemEvalSlot1CNNModel('SERVICE#GENERAL')
    model3 = SemEvalSlot1CNNModel('FOOD#QUALITY')
    model4 = SemEvalSlot1CNNModel('DRINKS#QUALITY')
    model5 = SemEvalSlot1CNNModel('DRINKS#STYLE_OPTIONS')
    model6 = SemEvalSlot1CNNModel('FOOD#PRICES')
    model7 = SemEvalSlot1CNNModel('DRINKS#PRICES')
    model8 = SemEvalSlot1CNNModel('RESTAURANT#MISCELLANEOUS')
    model9 = SemEvalSlot1CNNModel('FOOD#STYLE_OPTIONS')
    model10 = SemEvalSlot1CNNModel('LOCATION#GENERAL')
    model11 = SemEvalSlot1CNNModel('RESTAURANT#PRICES')
    model12 = SemEvalSlot1CNNModel('AMBIENCE#GENERAL')

    model0.train()
    model1.train()
    model2.train()
    model3.train()
    model4.train()
    model5.train()
    model6.train()
    model7.train()
    model8.train()
    model9.train()
    model10.train()
    model11.train()
    model12.train()

    with open("sem_eval_slot1_epoch_{}_hidden_{}.txt".format(n_epochs, n_hidden), "w") as f:
        index = 0
        for text_data, category_data in zip(text_data_lst[TRAINING_INDEX:], category_data_lst[TRAINING_INDEX:]):
            index += 1
            print(index, "/ 685")
            embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in text_data]
            for _ in range(max_sentence_len - len(embedded_sentence_data)):
                embedded_sentence_data.append([0.0] * 100)

            pred_lst = [model0.predict(embedded_sentence_data),
             model1.predict(embedded_sentence_data),
             model2.predict(embedded_sentence_data),
             model3.predict(embedded_sentence_data),
             model4.predict(embedded_sentence_data),
             model5.predict(embedded_sentence_data),
             model6.predict(embedded_sentence_data),
             model7.predict(embedded_sentence_data),
             model8.predict(embedded_sentence_data),
             model9.predict(embedded_sentence_data),
             model10.predict(embedded_sentence_data),
             model11.predict(embedded_sentence_data),
             model12.predict(embedded_sentence_data)]
            print("{0:<10} :".format("Predction"), pred_lst)
            f.write("".join([str(x) for x in pred_lst]) + "\n")

            # 중복 제거
            cat_lst = list(set(category_data))
            cat_index_lst = [CATEGORY_INDEX[cat] for cat in cat_lst]

            if len(cat_index_lst) == 0:
                #print("{0:<10} :".format("True set"), [1] + [0] * 12)
                f.write("1" + "0" * 12 + "\n")
            else:
                #print("{0:<10} :".format("True set"), [1 if x in cat_index_lst else 0 for x in range(13)])
                f.write("".join(["1" if x in cat_index_lst else "0" for x in range(13)]) + "\n")








