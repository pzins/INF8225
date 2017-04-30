from __future__ import print_function

import tensorflow as tf
import numpy as np
import datetime
now = datetime.datetime.now()


time = str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "h" + str(now.minute)

# import data
x_set = np.array([]).reshape(0, 32, 32, 3)
y_set = np.array([]).reshape(0, 2)
# ages = np.array([]).reshape(0, 1)
for it in range(1):
    x_tmp = np.load("data/x_32_" + str(it) + ".dat")
    y_tmp = np.load("data/y_32_" + str(it) + ".dat")
    x_set = np.append(x_set, x_tmp, axis=0)
    y_set = np.append(y_set, y_tmp, axis=0)
    
    # age = np.load("data1000/128_age/ytrain_128_" + str(it) + ".dat")
    # ages = np.append(ages, age)
# ages = np.expand_dims(ages, axis=1)

# create train, valid and test set
trainSize = int(x_set.shape[0] * 0.9)
validSize = int(x_set.shape[0] * 0.05)

x_train = x_set[:trainSize]
y_train = y_set[:trainSize]
x_val = x_set[trainSize:trainSize+validSize]
y_val = y_set[trainSize:trainSize+validSize]
x_test = x_set[trainSize+validSize:]
y_test = y_set[trainSize+validSize:]


# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 32
nb_batch = int(x_train.shape[0]/batch_size)
log_dir = "Tensorboard/" + time


# Network Parameters
img_size = 32
n_classes = 2
dropout = 1 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, img_size, img_size, 3])
    print(x.shape)
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv1 = maxpool2d(conv2, k=2)
    conv2 = conv2d(conv1, weights['wc3'], biases['bc3'])
    conv1 = conv2d(conv2, weights['wc4'], biases['bc4'])
    conv2 = maxpool2d(conv1, k=2)
    conv1 = tf.nn.dropout(conv2, dropout)

    # Fully connected layer
    fc1 = tf.reshape(conv1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    fc3 = tf.nn.dropout(fc3, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out


# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),

    'wc3': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64])),
    
    'wd1': tf.Variable(tf.random_normal([8*8*64, 100])),
    'wd2': tf.Variable(tf.random_normal([100, 100])),
    'wd3': tf.Variable(tf.random_normal([100, 100])),

    'out': tf.Variable(tf.random_normal([100, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([32])),

    'bc3': tf.Variable(tf.random_normal([64])),
    'bc4': tf.Variable(tf.random_normal([64])),
    
    'bd1': tf.Variable(tf.random_normal([100])),
    'bd2': tf.Variable(tf.random_normal([100])),
    'bd3': tf.Variable(tf.random_normal([100])),

    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# tensorboard summary
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("cost", cost)

# Initializing the variables
init = tf.global_variables_initializer()

# age distribution graph
# ages_distribution = tf.placeholder(tf.float32, [17097, 1] , name="ages")
# tf.summary.histogram("ages", ages_distribution )

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)    
    tf.global_variables_initializer().run()
    
    for epoch in range(training_epochs):
        for i in range(nb_batch):
            batch_x = x_train[i*batch_size:(i+1)*batch_size]
            batch_y = y_train[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop)
            _, summary = sess.run([optimizer, merged], feed_dict={x: batch_x, y: batch_y,
                                               keep_prob: dropout})
            train_writer.add_summary(summary, epoch * nb_batch + i)

        # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: x_val,
                                                          y: y_val,
                                                          keep_prob: 1.})
        print("Epoch (" + str(epoch) + ") Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    
    print("Optimization Finished!")
    res = sess.run(correct_pred, feed_dict={x: x_test, y: y_test, keep_prob: 1})
    loss, acc = sess.run([cost, accuracy], feed_dict={x: x_test,
                                                      y: y_test,
                                                      keep_prob: 1.})
    print(res)
    print("Accuracy : %f\n" % acc)