import tensorflow as tf
import xml_reader
from gensim.models import Word2Vec
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

# Global Variable
TRAINING_INDEX = 1315

# Data
input_token_data, output_label = xml_reader.load_data('data/ABSA16_Restaurants_Train_SB1_v2.xml')
sem_eval_restaurant_model = Word2Vec.load("sem_eval_restaurant_model.vec")
pos_tag = {"O": 1, "TARGET-B": 2, "TARGET-I": 3}
encoded_output_data = []
encoded_input_data = []

# NN Setting Variable
learning_rate = 0.0005
batch_size = 1
n_input = 100
num_classes = 4
hidden_size = 24
n_iteration = 10
max_sequence_len = 74

X = tf.placeholder(tf.float32, [None, max_sequence_len, n_input])
Y = tf.placeholder(tf.int32, [None, max_sequence_len])

cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, max_sequence_len, num_classes])

# Cost Function
W = tf.ones([batch_size, max_sequence_len])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=W)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

# Create training & test data
for tag_data in output_label:
    output_data = [pos_tag[tag] for tag in tag_data]

    # max_sequence 길이를 맞추기 위해 padding을 넣음
    for _ in range(max_sequence_len - len(output_data)):
        output_data.append(0)
    encoded_output_data.append(output_data)

for token_word_data in input_token_data:
    input_data = [list(sem_eval_restaurant_model.wv[word]) for word in token_word_data]

    # max_sequence 길이를 맞추기 위해 padding을 넣음
    for _ in range(max_sequence_len - len(input_data)):
        input_data.append([0.0] * 100)
    encoded_input_data.append(input_data)

train_input_data = encoded_input_data[:TRAINING_INDEX]
train_output_data = encoded_output_data[:TRAINING_INDEX]

test_input_data = encoded_input_data[TRAINING_INDEX:]
test_output_data = encoded_output_data[TRAINING_INDEX:]

# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):
        for input_data, output_data in zip(train_input_data, train_output_data):
            l, _ = sess.run([loss, train], feed_dict={X: [input_data], Y: [output_data]})

    with open("result.txt", "w") as f:
        for input_data, output_data in zip(test_input_data, test_output_data):
            result = sess.run(prediction, feed_dict={X: [input_data]})

            for ele in result[0]:
                f.write(str(ele))
            f.write("\n")

            for ele in output_data:
                f.write(str(ele))
            f.write("\n")
            f.write("-" * 20 + "\n")


