import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
import xml_reader


# Global Variable
TRAINING_INDEX = 1315

CATEGORY = ['NIL', 'RESTAURANT', 'SERVICE', 'FOOD', 'DRINKS', 'LOCATION', 'AMBIENCE']
DIVISION = ['GENERAL', 'QUALITY', 'STYLE_OPTIONS', 'PRICES', 'MISCELLANEOUS']

CATEGORY_DIVISION = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#QUALITY', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
                     'FOOD#PRICES', 'DRINKS#PRICES', 'RESTAURANT#MISCELLANEOUS', 'FOOD#STYLE_OPTIONS',
                     'LOCATION#GENERAL', 'RESTAURANT#PRICES', 'AMBIENCE#GENERAL']

CATEGORY_TRAIN_DATA = {
    'NIL': [],
    'RESTAURANT': [],
    'SERVICE': [],
    'FOOD': [],
    'DRINKS': [],
    'LOCATION': [],
    'AMBIENCE': []
}

DIVISION_TRAIN_DATA = {
    'GENERAL': [],
    'QUALITY': [],
    'STYLE_OPTIONS': [],
    'PRICES': [],
    'MISCELLANEOUS': [],
}

CATEGORY_INDEX = {
    'NIL': 0,
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

# Word Embedding
sem_eval_restaurant_model = Word2Vec.load("sem_eval_restaurant_model_add_yelp_aa.vec")

# Parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100
n_epochs = 100

max_sentence_len = 74
h_filter = 5  # height of filter
n_filter = 256
embed_vec_len = 100
n_hidden = 128  # for hidden dense layer
n_classes = 2

# Data
text_data_lst, category_data_lst = xml_reader.load_data_for_task_1_2()

for text_data, category_data in zip(text_data_lst[:TRAINING_INDEX], category_data_lst[:TRAINING_INDEX]):
    embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in text_data]

    for _ in range(max_sentence_len - len(embedded_sentence_data)):
        embedded_sentence_data.append([0.0] * 100)

    if len(category_data) == 0:
        CATEGORY_TRAIN_DATA["NIL"].append(embedded_sentence_data)
    else:
        for category in category_data:
            CATEGORY_TRAIN_DATA[category.split("#")[0]].append(embedded_sentence_data)
            DIVISION_TRAIN_DATA[category.split("#")[1]].append(embedded_sentence_data)


# Add train data
n_add_data = 300
ADD_TRAIN_DATA_YN = True

if ADD_TRAIN_DATA_YN:
    for cat in CATEGORY_DIVISION:
        with open("data/{}.data".format(cat)) as f:
            idx = 0

            for line in f.readlines():
                idx += 1
                if idx == n_add_data:
                    break
                if len(line.strip().split()) > 74:
                    continue

                embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in line.strip().split()]

                for _ in range(max_sentence_len - len(embedded_sentence_data)):
                    embedded_sentence_data.append([0.0] * 100)

                CATEGORY_TRAIN_DATA[cat.split("#")[0]].append(embedded_sentence_data)

    for cat in CATEGORY_DIVISION:
        with open("data/{}.data".format(cat)) as f:
            idx = 0

            for line in f.readlines():
                idx += 1
                if idx == n_add_data:
                    break
                if len(line.strip().split()) > 74:
                    continue

                embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in line.strip().split()]

                for _ in range(max_sentence_len - len(embedded_sentence_data)):
                    embedded_sentence_data.append([0.0] * 100)

                DIVISION_TRAIN_DATA[cat.split("#")[1]].append(embedded_sentence_data)


n_cat_total_data = 0
n_div_total_data = 0

for data in CATEGORY_TRAIN_DATA.values():
    n_cat_total_data += len(data)

for data in DIVISION_TRAIN_DATA.values():
    n_div_total_data += len(data)


# Model
class SemEvalSlot1CNNModel(object):
    def __init__(self, label, category_type):
        self.label = label
        self.cat_type = category_type

        # Session
        self.sess = tf.Session()

        # Data
        self.input_data = []
        self.output_data = []

        if category_type == "category":
            train_data_set = CATEGORY_TRAIN_DATA
        elif category_type == "division":
            train_data_set = DIVISION_TRAIN_DATA

        # Init data set
        for data_label, train_data in train_data_set.items():
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

        self.keep_prob = tf.placeholder(tf.float32)
        self.L2 = tf.nn.dropout(self.L2, self.keep_prob)

        # Layer 3 - FC
        self.W3 = tf.Variable(tf.random_normal([n_hidden, n_classes]))
        self.model = tf.matmul(self.L2, self.W3)

    def train(self):
        # Training
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Global Variable Initialize
        self.sess.run(tf.global_variables_initializer())

        if self.cat_type == "category":
            n_total_data = n_cat_total_data
        else:
            n_total_data = n_div_total_data

        total_batch = int(n_total_data / batch_size) + 1
        print("Total Batch:", total_batch)

        for epoch in range(n_epochs):
            for i in range(total_batch):
                index_start = i * batch_size
                index_end = min(n_total_data, (i + 1) * batch_size)
                batch_xs = self.input_data[index_start:index_end]
                batch_xs = np.expand_dims(batch_xs, -1)
                batch_ys = self.output_data[index_start:index_end]
                _, c = self.sess.run([optimizer, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.keep_prob: 0.7})
            print("Label : {}, Epoch: {}".format(self.label, epoch + 1))

    def predict(self, text_lst):
        text_lst = np.expand_dims(text_lst, -1)
        d = self.sess.run(self.model, feed_dict={self.X: [text_lst], self.keep_prob:1.0})
        return self.sess.run(tf.nn.softmax(d))[0][0] >= 0.2


if __name__ == "__main__":
    for cat, data in CATEGORY_TRAIN_DATA.items():
        print(cat, ":", len(data))

    print("-" * 30)

    for div, data in DIVISION_TRAIN_DATA.items():
        print(div, ":", len(data))

    print("-" * 30)

    nil_model = SemEvalSlot1CNNModel("NIL", "category")
    food_model = SemEvalSlot1CNNModel("FOOD", "category")
    restaurant_model = SemEvalSlot1CNNModel('RESTAURANT', "category")
    service_model = SemEvalSlot1CNNModel('SERVICE', "category")
    drinks_model = SemEvalSlot1CNNModel('DRINKS', "category")
    location_model = SemEvalSlot1CNNModel('LOCATION', "category")
    ambience_model = SemEvalSlot1CNNModel('AMBIENCE', "category")

    general_model = SemEvalSlot1CNNModel("GENERAL", "division")
    quality_model = SemEvalSlot1CNNModel('QUALITY', "division")
    style_options_model = SemEvalSlot1CNNModel('STYLE_OPTIONS', "division")
    prices_model = SemEvalSlot1CNNModel('PRICES', "division")
    miscellaneous_model = SemEvalSlot1CNNModel('MISCELLANEOUS', "division")

    nil_model.train()
    food_model.train()
    restaurant_model.train()
    service_model.train()
    drinks_model.train()
    location_model.train()
    ambience_model.train()

    general_model.train()
    quality_model.train()
    style_options_model.train()
    prices_model.train()
    miscellaneous_model.train()

    with open("result.txt".format(n_epochs, n_hidden, n_filter), "w") as f:
        index = 0
        for text_data, category_data in zip(text_data_lst[TRAINING_INDEX:], category_data_lst[TRAINING_INDEX:]):
            index += 1
            print(index, "/ 685")
            prediction_lst = [0] * 13
            embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in text_data]
            for _ in range(max_sentence_len - len(embedded_sentence_data)):
                embedded_sentence_data.append([0.0] * 100)

            if nil_model.predict(embedded_sentence_data):
                prediction_lst[0] = 1

            if food_model.predict(embedded_sentence_data):
                if quality_model.predict(embedded_sentence_data):
                    prediction_lst[3] = 1

                if prices_model.predict(embedded_sentence_data):
                    prediction_lst[6] = 1

                if style_options_model.predict(embedded_sentence_data):
                    prediction_lst[9] = 1

            if restaurant_model.predict(embedded_sentence_data):
                if general_model.predict(embedded_sentence_data):
                    prediction_lst[1] = 1

                if miscellaneous_model.predict(embedded_sentence_data):
                    prediction_lst[8] = 1

                if prices_model.predict(embedded_sentence_data):
                    prediction_lst[11] = 1

            if service_model.predict(embedded_sentence_data):
                prediction_lst[2] = 1

            if drinks_model.predict(embedded_sentence_data):
                if quality_model.predict(embedded_sentence_data):
                    prediction_lst[4] = 1

                if style_options_model.predict(embedded_sentence_data):
                    prediction_lst[5] = 1

                if prices_model.predict(embedded_sentence_data):
                    prediction_lst[7] = 1

            if location_model.predict(embedded_sentence_data):
                prediction_lst[10] = 1

            if ambience_model.predict(embedded_sentence_data):
                prediction_lst[12] = 1

            if sum(prediction_lst) == 0:
                prediction_lst[0] = 1

            print("{0:<10} :".format("Predction"), prediction_lst)
            f.write("".join([str(x) for x in prediction_lst]) + "\n")

            # 중복 제거
            cat_lst = list(set(category_data))
            cat_index_lst = [CATEGORY_INDEX[cat] for cat in cat_lst]

            if len(cat_index_lst) == 0:
                print("{0:<10} :".format("True set"), [1] + [0] * 12)
                f.write("1" + "0" * 12 + "\n")
            else:
                print("{0:<10} :".format("True set"), [1 if x in cat_index_lst else 0 for x in range(13)])
                f.write("".join(["1" if x in cat_index_lst else "0" for x in range(13)]) + "\n")

