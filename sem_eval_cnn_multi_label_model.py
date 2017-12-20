import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import xml_reader
from nltk.tag import StanfordNERTagger

# Global Variable
TRAINING_INDEX = 1315 
VALIDATION_INDEX = 1315
CATEGORY =['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#QUALITY', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
           'FOOD#PRICES', 'DRINKS#PRICES', 'RESTAURANT#MISCELLANEOUS', 'FOOD#STYLE_OPTIONS', 'LOCATION#GENERAL',
           'RESTAURANT#PRICES', 'AMBIENCE#GENERAL']

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

TRAIN_DATA = {
    'NIL': [],
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
sem_eval_restaurant_model = Word2Vec.load("sem_eval_restaurant_model_add_data.vec")
prob_data = KeyedVectors.load_word2vec_format('word_naive_bayes_probability.txt')
#sem_eval_restaurant_model = KeyedVectors.load_word2vec_format('wiki.simple.vec')

# NER Tag
st = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')
NER_dict = {
'O' : 				[0, 0, 0, 0, 0, 0, 0],
'PERSON' : 			[1, 0, 0, 0, 0, 0, 0],
'LOCATION' : 		[0, 1, 0, 0, 0, 0, 0],
'ORGANIZATION' : 	[0, 0, 1, 0, 0, 0, 0],
#'MISC' : 			[0, 0, 0, 1]
'MONEY' : 			[0, 0, 0, 1, 0, 0, 0],
'PERCENT' :  		[0, 0, 0, 0, 1, 0, 0],
'DATE' :  			[0, 0, 0, 0, 0, 1, 0], 
'TIME' :  			[0, 0, 0, 0, 0, 0, 1]
}


# Parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100
n_epochs = 100
n_total_data = 0

max_sentence_len = 74
h_filter = 5  # height of filter
n_filter = 256
embed_vec_len = 100
prob_len = 13
n_hidden = 128  # for hidden dense layer
n_classes = 2
threshold = 0.2
keep_prob = 0.5

# Data
text_data_lst, category_data_lst = xml_reader.load_data_for_task_1_2()

for text_data, category_data in zip(text_data_lst[:TRAINING_INDEX], category_data_lst[:TRAINING_INDEX]):
    #embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in text_data]

#    tagged = st.tag(text_data);
    tagged = ['\0'] * len(text_data)
    embedded_sentence_data = []

    for word, tag in zip(text_data, tagged) :
        try : 
            embedded_sentence_data.append(list(sem_eval_restaurant_model.wv[word.lower()]))
        except KeyError:
            embedded_sentence_data.append([0.0] * embed_vec_len)
        embedded_sentence_data[-1] += list(prob_data.wv[word.lower()])
#        embedded_sentence_data[-1] += NER_dict[tag[1]]
   

    for _ in range(max_sentence_len - len(embedded_sentence_data)):
        embedded_sentence_data.append([0.0] * (embed_vec_len + prob_len))

    if len(category_data) == 0:
        TRAIN_DATA["NIL"].append(embedded_sentence_data)
    else:
        for category in category_data:
            TRAIN_DATA[category].append(embedded_sentence_data)


# Add train data
n_add_data = 100

for cat in CATEGORY:
    with open("data/{}.data".format(cat)) as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            if idx == n_add_data:
                break
            if len(line.strip().split()) > 74:
                continue

            #embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in line.strip().split()]
#            tagged = st.tag(line.strip().split())
            tagged = ['\0'] * len(line.strip().split())
            embedded_sentence_data = []
            for word, tag in zip(line.strip().split(), tagged) : 
                try:
                    embedded_sentence_data.append(list(sem_eval_restaurant_model.wv[word.lower()]))
                except KeyError:
                    embedded_sentence_data.append([0.0] * embed_vec_len)
                embedded_sentence_data[-1] += list(prob_data.wv[word.lower()])
#                embedded_sentence_data[-1] += NER_dict[tag[1]]
        

            for _ in range(max_sentence_len - len(embedded_sentence_data)):
                embedded_sentence_data.append([0.0] * (embed_vec_len + prob_len))

            TRAIN_DATA[cat].append(embedded_sentence_data)

for data in TRAIN_DATA.values():
    n_total_data += len(data)


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
        self.X = tf.placeholder(tf.float32, [None, max_sentence_len, embed_vec_len + prob_len, 1])
        self.Y = tf.placeholder(tf.float32, [None, 2])

        # Layer 1 - CNN
        self.W1 = tf.Variable(tf.random_normal([h_filter, embed_vec_len + prob_len, 1, n_filter], stddev=0.01))

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

        total_batch = int(n_total_data / batch_size) + 1
        print("Total Batch:", total_batch)
        
#        rights=[]
        for epoch in range(n_epochs):
            for i in range(total_batch):
                index_start = i * batch_size
                index_end = min(n_total_data, (i + 1) * batch_size)
                batch_xs = self.input_data[index_start:index_end]
                batch_xs = np.expand_dims(batch_xs, -1)
                batch_ys = self.output_data[index_start:index_end]
                _, c = self.sess.run([optimizer, cost], feed_dict={self.X: batch_xs, self.Y: batch_ys, self.keep_prob: keep_prob})

#            right = 0
            #validation to stop loop
#            for text_data, category_data in zip(text_data_lst[TRAINING_INDEX:VALIDATION_INDEX], category_data_lst[TRAINING_INDEX:VALIDATION_INDEX]):
#                embedded_sentence_data = []
#               for word in text_data : 
#                    try:
#                        embedded_sentence_data.append(list(sem_eval_restaurant_model.wv[word.lower()]))
#                    except KeyError:
#                        embedded_sentence_data.append([0.0] * embed_vec_len)
#
#                for _ in range(max_sentence_len - len(embedded_sentence_data)):
#                    embedded_sentence_data.append([0.0] * embed_vec_len)
#		
#                text_lst = np.expand_dims(embedded_sentence_data, -1)
#                d=self.sess.run(self.model, feed_dict={self.X: [text_lst]})
#                if self.sess.run(tf.nn.softmax(d))[0][0] >= threshold:
#                    right += 1
#            rights.append(right)

            print("Label : {}, Epoch: {}".format(self.label, epoch + 1))
#            if epoch > 3 and abs(rights[epoch-3]-right)<3 and abs(rights[epoch-2]-right)<3 and (rights[epoch-1]-right)<3 :
#                break

    def predict(self, text_lst):
        text_lst = np.expand_dims(text_lst, -1)
        d = self.sess.run(self.model, feed_dict={self.X: [text_lst], self.keep_prob:1.0})
        if self.sess.run(tf.nn.softmax(d))[0][0] >= threshold:
            return 1
        else:
            return 0


if __name__ == "__main__":
    model0 = SemEvalSlot1CNNModel('NIL')
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

    with open("sem_eval_slot1_epoch_{}_hidden_{}_filter_{}_adam_select_200_batch_50_em_nil.txt".format(n_epochs, n_hidden, n_filter), "w") as f:
        index = 0
        for text_data, category_data in zip(text_data_lst[TRAINING_INDEX:], category_data_lst[TRAINING_INDEX:]):
            index += 1
            print(index, "/ 685")
            #embedded_sentence_data = [list(sem_eval_restaurant_model.wv[word]) for word in text_data]
#            tagged = st.tag(text_data)
            tagged = ['\0'] * len(text_data)
            embedded_sentence_data = []
            for word, tag in zip(text_data, tagged) : 
                try:
                    embedded_sentence_data.append(list(sem_eval_restaurant_model.wv[word.lower()]))
                except KeyError:
                    embedded_sentence_data.append([0.0] * embed_vec_len)
                embedded_sentence_data[-1] += list(prob_data.wv[word.lower()])
#                embedded_sentence_data[-1] += NER_dict[tag[1]]

            for _ in range(max_sentence_len - len(embedded_sentence_data)):
                embedded_sentence_data.append([0.0] * (embed_vec_len + prob_len))

            pred_lst = [
                model0.predict(embedded_sentence_data),
                #0,
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
                model12.predict(embedded_sentence_data)
            ]

            if sum(pred_lst) == 0:
                pred_lst[0] = 1

            print("{0:<10} :".format("Predction"), pred_lst)
            f.write("".join([str(x) for x in pred_lst]) + "\n")

            # 중복 제거
            cat_lst = list(set(category_data))
            cat_index_lst = [CATEGORY_INDEX[cat] for cat in cat_lst]

            if len(cat_index_lst) == 0:
                print("{0:<10} :".format("True set"), [1] + [0] * 12)
                f.write("1" + "0" * 12 + "\n")
            else:
                print("{0:<10} :".format("True set"), [1 if x in cat_index_lst else 0 for x in range(13)])
                f.write("".join(["1" if x in cat_index_lst else "0" for x in range(13)]) + "\n")
