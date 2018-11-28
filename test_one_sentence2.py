from Cython.Shadow import inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import util
import numpy as np
import tensorflow as tf
wordsList = np.load('./training_data/wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('./training_data/wordVectors.npy')
file_index = 1

#docmatrix = np.load('./training_data/idsMatrix.npy')
#user_data_path = util.get_file_path('./user_data')
#print(user_data_path)
fill_null = []
for i in range(24):
    fill_null.append([1,0])
#user_data_path = util.get_file_path('./user_data')
user_comment = np.zeros((250), dtype='int32')

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 50000
numDimensions = 300
maxSeqLength = 250

tf.reset_default_graph()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

#the parameter we need to compute by the neural network
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('./model_path1'))

with open('./comment.txt') as f:
    #print(user_data_path[1])
    indexCounter = 0
    line = f.readline()
    cleanedLine = util.cleanSentences(line)
    #print(cleanedLine)
    split = cleanedLine.split()
    for word in split:
        try:
            user_comment[indexCounter] = wordsList.index(word)
        except ValueError:
            user_comment[indexCounter] = 399999  # Vector for unknown words
        indexCounter = indexCounter + 1

ex_add = np.zeros(shape=(1, 250))
# print(ex_add)
user_comment = user_comment[np.newaxis, :]
# print(user_comment[0][0])
for i in range(23):
    user_comment = np.vstack([user_comment, ex_add])
#print(type(user_comment))
pred = sess.run(prediction, {input_data: user_comment, labels: fill_null})
# print(pred)
pred_cls = pred
pred_cls = np.argmax(pred_cls, axis=1)
#print(pred_cls[0])
if (pred_cls[0] == 0):
    print('thanks')
else:
    print('sorry')