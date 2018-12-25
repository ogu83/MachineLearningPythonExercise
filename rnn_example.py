import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
# from tensorflow.contrib.rnn.python.ops import rnn, rnn_cell
# from tensorflow.contrib.rnn import BasicLSTMCell, static_rnn


mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
 
hm_epochs = 3
n_classes = 10
batch_size = 128
chunk_size=28
n_chunks=28
rnn_size=128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def reccurent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    print(x)
    x = tf.transpose(x, [1,0,2])
    print(x)
    x = tf.reshape(x, [-1, chunk_size])
    # x = tf.split(0, n_chunks, x)
    x = tf.split(x, n_chunks, 0)
    print(x)

    # lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    # outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    # outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    print(x)

    return output

def train_neural_network(x):
    prediction = reccurent_neural_network(x)
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=prediction,logits=y) )
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(y, tf.shape(prediction)), logits=prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())        

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)         
                epoch_x = epoch_x.reshape(batch_size,n_chunks,chunk_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape(-1, n_chunks, chunk_size), y:mnist.test.labels}))

if __name__ == "__main__":
    train_neural_network(x)