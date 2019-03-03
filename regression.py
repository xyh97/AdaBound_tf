import os
import model
import tensorflow as tf
import numpy as np
from adabound_tf import AdaBoundOptimizer
import random

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
train_size = y_train.shape[0]
total_epoch = 100
batch_size = 128
init_lr = 0.001
init_final_lr = 0.01
init_gamma = 1e-3
flag = (train_size%batch_size==0)
num_batch = train_size//batch_size if flag else (train_size//batch_size + 1)



# model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 28, 28])
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
final_lr = tf.placeholder(tf.float32, name='final_lr')
gamma = tf.placeholder(tf.float32, name='gamma')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
train_step = AdaBoundOptimizer(learning_rate=learning_rate, final_lr=final_lr, beta1=0.9, beta2=0.999,
                               gamma=gamma, amsbound=False).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    epoch_lr = init_lr
    epoch_final_lr = init_final_lr
    for epoch in range(total_epoch):
        print('Epoch: %d' % epoch)
        # shuffle training data
        indices = list(range(train_size))
        random.shuffle(indices)
        x_train_epoch, y_train_epoch = x_train[indices], y_train_one_hot[indices]

        # learning rate decay at epoch 150
        if epoch == 150:
            epoch_lr = epoch_lr/10
            epoch_final_lr = epoch_final_lr/10

        # batch training
        epoch_loss = 0
        for i in range(num_batch):
            step += 1
            gamma_t = init_gamma * step
            if flag or i!=num_batch-1:
                batch_xs, batch_ys = x_train_epoch[i*batch_size:(i+1)*batch_size], y_train_epoch[i*batch_size:(i+1)*batch_size]
            else:
                batch_xs, batch_ys = x_train_epoch[i*batch_size:], y_train_epoch[i*batch_size:]
            train_feed_dict = {
                x: batch_xs,
                y_: batch_ys,
                learning_rate: epoch_lr,
                final_lr: epoch_final_lr,
                gamma: gamma_t,
            }
            _, loss = sess.run([train_step, cross_entropy], feed_dict=train_feed_dict)
            epoch_loss += loss * len(batch_xs)
        epoch_loss = epoch_loss/train_size
        print('train loss %.3f' % epoch_loss)
        test_acc = sess.run(accuracy, feed_dict={x: x_test, y_: y_test_one_hot})
        print('test acc %.3f\n' % test_acc)

    # path = saver.save(
    #     sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
    #     write_meta_graph=False, write_state=False)
    # print("Saved:", path)