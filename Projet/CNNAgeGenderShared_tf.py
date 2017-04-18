from __future__ import print_function

import tensorflow as tf
import numpy as np
import datetime
now = datetime.datetime.now()
import keras


time = str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.hour) + "h" + str(now.minute)


num_classes_gender = 2

def getAgeCategory(age):  
  return age

x_set = np.array([]).reshape(0, 128, 128, 3)
y_set_age = np.array([]).reshape(0,2)
y_set_gender = np.array([]).reshape(0,2)
for it in range(1):
    x_tmp = np.load("data/x_128_" + str(it) + ".dat")
    y_tmp = np.load("data/y_128_" + str(it) + ".dat")
    x_set = np.append(x_set, x_tmp, axis=0)
    y_set_age = np.append(y_set_age, y_tmp, axis=0)
    y_set_gender = np.append(y_set_gender, y_tmp, axis=0)

y_set_age = np.delete(y_set_age, 0, 1)
y_set_gender = np.delete(y_set_gender, -1, 1)

for i in range(len(y_set_age)):
  y_set_age[i] = getAgeCategory(y_set_age[i])

y_set_gender = keras.utils.to_categorical(y_set_gender, num_classes_gender)

y_set = np.column_stack((y_set_age, y_set_gender))


trainSize = int(x_set.shape[0] * 0.9)
validSize = int(x_set.shape[0] * 0.05)

x_train = x_set[:trainSize]
y_train_gender = y_set_gender[:trainSize]
y_train_age = y_set_age[:trainSize]

x_val = x_set[trainSize:trainSize+validSize]
y_val_age = y_set_age[trainSize:trainSize+validSize]
y_val_gender = y_set_gender[trainSize:trainSize+validSize]

x_test = x_set[trainSize+validSize:]
y_test_age = y_set_age[trainSize+validSize:]
y_test_gender = y_set_gender[trainSize+validSize:]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 32
nb_batch = int(x_train.shape[0]/batch_size)
log_dir = "Tensorboard/" + time


# Network Parameters
img_size = 128
n_classes = 1
dropout = 0.5 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
y_age = tf.placeholder(tf.float32, [None, 1])
y_gender = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

y = y_age

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
    x = maxpool2d(x, k=4)
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
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 32])),
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
cost = tf.reduce_mean(tf.square(pred - y))
accuracy = tf.reduce_mean(tf.abs(tf.subtract(pred, y)))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
correct_pred_age = pred#tf.abs(pred - y)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# tensorboard summary
tf.summary.scalar("accuracy", accuracy)
tf.summary.scalar("cost", cost)

# Initializing the variables
init = tf.global_variables_initializer()

# age distribution graph
# ages_distribution = tf.placeholder(tf.float32, [17097, 1] , name="ages")
# tf.summary.histogram("ages", ages_distribution )

batch_x_val = x_val[:32]
batch_y_val = y_val_age[:32]
x_test = x_test[:128]
y_test_age = y_test_age[:128]

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
            batch_y = y_train_age[i*batch_size:(i+1)*batch_size]
            # Run optimization op (backprop)
            _, summary = sess.run([optimizer, merged], feed_dict={x: batch_x, y: batch_y,
                                               keep_prob: dropout})
            train_writer.add_summary(summary, epoch * nb_batch + i)

        # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x_val,
                                                      y: batch_y_val,
                                                      keep_prob: 1.})
        res = sess.run(pred, feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1})
        print(res)
        print("Epoch (" + str(epoch) + ") Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
    print("Optimization Finished!")
    # loss, acc = sess.run([cost, accuracy], feed_dict={x: x_train[:100],
                                                      # y: y_train_age[:100],
                                                      # keep_prob: 1.})
    # for i in range(len(res)):
        # print(str(res[i]) + " -> " + str(y_test_age[i]))
    # print("Accuracy : %f\n" % acc)