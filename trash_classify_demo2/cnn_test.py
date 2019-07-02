import tensorflow as tf
import linecache
import cv2
import numpy as np
import time

train_images_path = 'train_dataset/'
train_labels_path = 'train_label.txt'


def load_train_dataset(index):  # 从1开始
    line_str = linecache.getline(train_labels_path, index)
    image_name, image_label = line_str.split(' ')
    image = cv2.imread(train_images_path + image_name)
    image = np.array(image, dtype=np.float32) / 255
    return image, [image_label]


def combine_train_dataset(count, size):
    train_images_load = np.zeros(shape=(size, 200, 200, 3))
    train_labels_load = np.zeros(shape=(size, 4))
    for i in range(count, size):
        train_images_load[i], train_labels_load[i] = load_train_dataset(i + 1)
    count += size
    return train_images_load, train_labels_load, count


# 通过L2正则化防止过拟合
def weight_variable_with_loss(shape, stddev, lam):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if lam is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weight), lam, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return weight


def bias_variable(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def lrn(x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# 输入层
with tf.name_scope('input_layer'):
    x_input = tf.placeholder(tf.float32, [None, 200, 200, 3])
    y_input = tf.placeholder(tf.float32, [None, 4])
    keep_prob = tf.placeholder(tf.float32)

# 第一个卷积层
# 卷积核[5, 5, 3, 64]
with tf.name_scope('conv1_layer'):
    w_conv1 = weight_variable_with_loss([5, 5, 3, 64], stddev=5e-2, lam=0)
    b_conv1 = bias_variable([64])
    conv_kernel1 = conv2d(x_input, w_conv1)
    conv1 = tf.nn.relu(tf.nn.bias_add(conv_kernel1, b_conv1))
    pool1 = max_pool_2x2(conv1)  # 200*200 -> 100*100
    result1 = pool1

# 第二个卷积层
# 卷积核[5, 5, 64, 64]
with tf.name_scope('conv2_layer'):
    w_conv2 = weight_variable_with_loss([5, 5, 64, 64], stddev=5e-2, lam=0)
    b_conv2 = bias_variable([64])
    conv_kernel2 = conv2d(result1, w_conv2)
    conv2 = tf.nn.relu(tf.nn.bias_add(conv_kernel2, b_conv2))
    pool2 = max_pool_2x2(conv2)  # 100*100 -> 50*50
    result2 = pool2

# 第三个卷积层
# 卷积核[5, 5, 64, 64]
with tf.name_scope('conv3_layer'):
    w_conv3 = weight_variable_with_loss([5, 5, 64, 64], stddev=5e-2, lam=0)
    b_conv3 = bias_variable([64])
    conv_kernel3 = conv2d(result2, w_conv3)
    conv3 = tf.nn.relu(tf.nn.bias_add(conv_kernel3, b_conv3))
    pool3 = max_pool_2x2(conv3)  # 50*50 -> 25*25
    result3 = pool3

# 第一个全连接层
with tf.name_scope('fc4_layer'):
    w_fc4 = weight_variable_with_loss([25 * 25 * 64, 512], stddev=5e-2, lam=0.004)
    b_fc4 = bias_variable([512])
    result3_flat = tf.reshape(result3, [-1, 25 * 25 * 64])
    fc4 = tf.nn.relu(tf.matmul(result3_flat, w_fc4) + b_fc4)
    result4 = tf.nn.dropout(fc4, keep_prob)

# 第二个全连接层
with tf.name_scope('fc5_layer'):
    w_fc5 = weight_variable_with_loss([512, 256], stddev=5e-2, lam=0.004)
    b_fc5 = bias_variable([256])
    fc5 = tf.nn.relu(tf.matmul(result4, w_fc5) + b_fc5)
    result5 = tf.nn.dropout(fc5, keep_prob)

# 输出层
with tf.name_scope('output_layer'):
    w_fc6 = weight_variable_with_loss([256, 4], stddev=1 / 256.0, lam=0)
    b_fc6 = bias_variable([4])
    fc6 = tf.nn.softmax(tf.matmul(fc5, w_fc6) + b_fc6)
    logits = fc6

# 损失函数
with tf.name_scope('loss'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y_input, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'))

# 训练函数
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 计算准确率
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 会话初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
save_dir = "classify_modles"
checkpoint_name = "train.ckpt"

# 变量初始化
training_steps = 50
display_step = 5
batch_size = 50
train_images_count = 0

# 训练
print('Start training...')

for step in range(training_steps):
    train_images, train_labels, train_images_count = combine_train_dataset(train_images_count, batch_size)
    train_step.run(feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 0.5})
    if step % display_step == 0:
        train_accuracy = accuracy.eval(feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 1.0})
        train_loss = sess.run(loss, feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 0.5})
        print("step {}\n training accuracy {}\n loss {}".format(step, train_accuracy, train_loss))

sess.close()