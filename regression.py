import os
import model
import tensorflow as tf
import numpy as np
from adabound_tf import AdaBoundOptimizer

def load_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data('./MNIST_data//mnist.npz')
y_train_one_hot = np.zeros((y_train.shape[0], 10))
y_test_one_hot = np.zeros((y_test.shape[0], 10))
y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1



# model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 28, 28])
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
gamma = tf.placeholder(tf.float32, name='gamma')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
train_step = AdaBoundOptimizer(learning_rate=0.001, final_lr=0.01, beta1=0.9, beta2=0.999,
                               gamma=gamma, amsbound=False).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        g = 1e-3*1
        batch_xs, batch_ys = x_train[i*100:i*100+100], y_train_one_hot[i*100:i*100+100]
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys, gamma: g})
        print(loss)

    print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test_one_hot}))

    # path = saver.save(
    #     sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
    #     write_meta_graph=False, write_state=False)
    # print("Saved:", path)