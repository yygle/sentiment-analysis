from Cython.Shadow import inline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import util
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
wordsList = np.load('./training_data/wordsList.npy')
#wordsList = np.load('H:/python_space/NLP-YYG/word2vec/Gensim-w2v/text8.eng-try.text.model')
#print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('./training_data/wordVectors.npy')
#wordVectors = np.load('H:/python_space/NLP-YYG/word2vec/Gensim-w2v/text8.eng-try.text.model.wv.vectors.npy')
#print ('Loaded the word vectors!')
# this part is to definite the parameters we need in the project
def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdata_path', type=str, default='./training_data/positiveReviews/',
                        help='data folder contains all postive comments')
    parser.add_argument('--ndata_path', type=str, default='./training_data/negativeReviews/',
                        help='data folder contains all postive comments')
    parser.add_argument('--data_info', type=str, default='False',
                        help='choose whether to show the infomation of dataset')
    parser.add_argument('--isTrainingW2v', type=str, default='False',
                        help='choose whether to train the word embedding part')
    parser.add_argument('--isTrainingD2v', type=str, default='False',
                        help='choose whether to train document to vectors')
    parser.add_argument('--isTrainingPredModel', type=str, default='False',
                        help='choose whether to train the prediction model')
    parser.add_argument('--numDimensions', type=int, default=300,
                        help='dimensions of the embedding')
    parser.add_argument('--batchSize', type=int, default=24,
                        help='size of the data feeded in lstm')
    parser.add_argument('--lstmUnits', type=int, default=64,
                        help='number of cells of lstm')
    parser.add_argument('--numClasses', type=int, default=2,
                        help='minibatch size')
    parser.add_argument('--iterations', type=int, default=50000,
                        help='iterations to build the model')
    parser.add_argument('--model_test_iters', type=int, default=10,
                        help='iterations to test the accuracy of the model')
    return parser
parser = get_argument_parser()
parameter = parser.parse_args()
#the part is added in case we do not have enough gpu memory
'''
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config.gpu_options.allow_growth = True
sess = tf.Session()
'''
#the part is to load files
positiveFiles = util.get_file_path('./training_data/positiveReviews/')
negativeFiles = util.get_file_path('./training_data/negativeReviews/')
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
#print('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        counter = len(line.split())
        numWords.append(counter)
#print('Negative files finished')

numFiles = len(numWords)
if parameter.data_info == 'True':
    print('The total number of files is', numFiles)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', sum(numWords) / len(numWords))

    #% matplotlib  inline
    plt.hist(numWords, 50)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.axis([0, 1200, 0, 8000])
    plt.show()


#the part is to build the document to the index of vectors
maxSeqLength = 250
if parameter.isTrainingD2v == 'True':
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    for pf in positiveFiles:
        with open(pf, "r") as f:
            indexCounter = 0
            line = f.readline()
            cleanedLine = util.cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            fileCounter = fileCounter + 1

    for nf in negativeFiles:
        with open(nf, "r") as f:
            indexCounter = 0
            line = f.readline()
            cleanedLine = util.cleanSentences(line)
            split = cleanedLine.split()
            for word in split:
                try:
                    ids[fileCounter][indexCounter] = wordsList.index(word)
                except ValueError:
                    ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
                indexCounter = indexCounter + 1
                if indexCounter >= maxSeqLength:
                    break
            fileCounter = fileCounter + 1
            # Pass into embedding function and see if it evaluates.

    np.save('idsMatrix', ids)
docmatrix = np.load('./training_data/idsMatrix.npy')


#this part is to build the lstm model
batchSize = parameter.batchSize
lstmUnits = parameter.lstmUnits
numClasses = parameter.numClasses
iterations = parameter.iterations
numDimensions = parameter.numDimensions

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

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


# this part is to train the prdiction model
if(parameter.isTrainingPredModel == 'True'):
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        # Next Batch of reviews
        nextBatch, nextBatchLabels = util.getTrainBatch(docmatrix,batchSize,maxSeqLength)
        sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

        if (i % 1000 == 0 and i != 0):
            loss_ = sess.run(loss, {input_data: nextBatch, labels: nextBatchLabels})
            accuracy_ = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})

            print("iteration {}/{}...".format(i + 1, iterations),
                  "loss {}...".format(loss_),
                  "accuracy {}...".format(accuracy_))
            # Save the network every 10,000 training iterations
        if (i % 10000 == 0 and i != 0):
            save_path = saver.save(sess, "model_path/pretrained_lstm_model.ckpt", global_step=i)
            print("saved to %s" % save_path)

else:
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./model_path'))
    user_data_path = util.get_file_path('./user_data')
    # print(user_data_path)
    nextBatch, nextBatchLabels = util.getTestBatch(docmatrix,batchSize,maxSeqLength)
    #print(nextBatchLabels[0])
    user_comment = np.zeros((maxSeqLength), dtype='int32')
    with open(user_data_path[1]) as f:
        indexCounter = 0
        line = f.readline()
        cleanedLine = util.cleanSentences(line)
        # print(cleanedLine)
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
    # print(user_comment)
    pred = sess.run(prediction, {input_data: user_comment, labels: nextBatchLabels})
    # print(pred)
    pred_cls = pred
    pred_cls = np.argmax(pred_cls, axis=1)
    print(pred_cls[0])
'''
    if (pred_cls[0] == 0):
        print('thanks')
    else:
        print('sorry')
'''

