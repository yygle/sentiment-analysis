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
for i in range(20):
    fill_null.append([1,0])
#user_data_path = util.get_file_path('./user_data')
#user_comment = np.zeros((30), dtype='int32')

user_comment = np.array([399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999,399999])

batchSize = 20
lstmUnits = 64
numClasses = 2
iterations = 2000
numDimensions = 300
maxSeqLength = 30

tf.reset_default_graph()
sess = tf.InteractiveSession()
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

with tf.name_scope('data'):
    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

#the parameter we need to compute by the neural network
with tf.name_scope('weight'):
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
with tf.name_scope('bias'):
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
with tf.name_scope('value'):
    value = tf.transpose(value, [1, 0, 2])
with tf.name_scope('last'):
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
with tf.name_scope('y_prediction'):
    prediction = (tf.matmul(last, weight) + bias)
with tf.name_scope('correctPred'):
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    #tf.summary.scalar('accuracy', accuracy)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer().minimize(loss)

#sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint("./short_model"))

with open('./comment.txt') as f:
    #print(user_data_path[1])
    indexCounter = 0
    line = f.readline()
    cleanedLine = util.cleanSentences(line)
    print(cleanedLine)
    split = cleanedLine.split()
    for word in split:
        try:
            user_comment[indexCounter] = wordsList.index(word)
        except ValueError:
            user_comment[indexCounter] = 399999  # Vector for unknown words
        indexCounter = indexCounter + 1

ex_add = np.zeros(shape=(1, 30))
# print(ex_add)
user_comment = user_comment[np.newaxis, :]
# print(user_comment[0][0])
for i in range(19):
    user_comment = np.vstack([user_comment, ex_add])
#print(user_comment.shape)
pred = sess.run(prediction, {input_data: user_comment, labels: fill_null})
# print(pred)
pred_cls = pred
pred_cls = np.argmax(pred_cls, axis=1)
#print(pred_cls[0])
if (pred_cls[0] == 0):
    print('thanks')
else:
    print('sorry')