import tensorflow as tf
import linecache
import cv2
import numpy as np
import os

train_images_path = 'D:/WorkSpace/Python/trash_classify_dataset/dataset/'
train_labels_path = 'D:/WorkSpace/Python/trash_classify_dataset/train_label.txt'
test_images_path = 'D:/WorkSpace/Python/trash_classify_dataset/dataset/'
test_labels_path = 'D:/WorkSpace/Python/trash_classify_dataset/test_label.txt'

classify_num = 50
train_images_num = 29081
test_images_num = 3232


def load_train_dataset(index):  # 从1开始
    if index > train_images_num:
        if index % train_images_num == 0:
            index = train_images_num
        else:
            index %= train_images_num
    line_str = linecache.getline(train_labels_path, index)
    image_name, image_label = line_str.split(' ')
    image = cv2.imread(train_images_path + image_name)
    # cv2.imshow('pic',image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image, image_label


def combine_train_dataset(count, size):
    train_images_load = np.zeros(shape=(size, 224, 224, 3))
    train_labels_load = np.zeros(shape=(size, classify_num))
    for i in range(size):
        train_images_load[i], train_labels_index = load_train_dataset(count + i + 1)
        train_labels_load[i][int(train_labels_index) - 1] = 1.0
    count += size
    return train_images_load, train_labels_load, count


def load_test_dataset(index):  # 从1开始
    if index > test_images_num:
        if index % test_images_num == 0:
            index = test_images_num
        else:
            index %= test_images_num
    line_str = linecache.getline(test_labels_path, index)
    image_name, image_label = line_str.split(' ')
    image = cv2.imread(test_images_path + image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    return image, image_label


def combine_test_dataset(count, size):
    test_images_load = np.zeros(shape=(size, 224, 224, 3))
    test_labels_load = np.zeros(shape=(size, classify_num))
    for i in range(size):
        test_images_load[i], test_labels_index = load_test_dataset(count + i + 1)
        test_labels_load[i][int(test_labels_index) - 1] = 1.0
    count += size
    return test_images_load, test_labels_load, count


def weight_variable(shape):
    weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return weight


def bias_variable(shape):
    bias = tf.Variable(tf.constant(0.0, shape=shape))
    return bias


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 输入层
with tf.name_scope('input_layer'):
    x_input = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_input = tf.placeholder(tf.float32, [None, classify_num])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
	# 数据集平均RGB值
    mean = tf.constant([159.780, 139.802, 119.047], dtype=tf.float32, shape=[1, 1, 1, 3])
    x_input = x_input - mean

# 第一个卷积层 size:224
# 卷积核1[3, 3, 3, 64]
# 卷积核2[3, 3, 64, 64]
with tf.name_scope('conv1_layer'):
    w_conv1 = weight_variable([3, 3, 3, 64])
    b_conv1 = bias_variable([64])
    conv_kernel1 = conv2d(x_input, w_conv1)
    bn1 = tf.layers.batch_normalization(conv_kernel1, training=is_training)
    conv1 = tf.nn.relu(tf.nn.bias_add(bn1, b_conv1))

    w_conv2 = weight_variable([3, 3, 64, 64])
    b_conv2 = bias_variable([64])
    conv_kernel2 = conv2d(conv1, w_conv2)
    bn2 = tf.layers.batch_normalization(conv_kernel2, training=is_training)
    conv2 = tf.nn.relu(tf.nn.bias_add(bn2, b_conv2))

    pool1 = max_pool_2x2(conv2)  # 224*224 -> 112*112
    result1 = pool1

# 第二个卷积层 size:112
# 卷积核3[3, 3, 64, 128]
# 卷积核4[3, 3, 128, 128]
with tf.name_scope('conv2_layer'):
    w_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    conv_kernel3 = conv2d(result1, w_conv3)
    bn3 = tf.layers.batch_normalization(conv_kernel3, training=is_training)
    conv3 = tf.nn.relu(tf.nn.bias_add(bn3, b_conv3))

    w_conv4 = weight_variable([3, 3, 128, 128])
    b_conv4 = bias_variable([128])
    conv_kernel4 = conv2d(conv3, w_conv4)
    bn4 = tf.layers.batch_normalization(conv_kernel4, training=is_training)
    conv4 = tf.nn.relu(tf.nn.bias_add(bn4, b_conv4))

    pool2 = max_pool_2x2(conv4)  # 112*112 -> 56*56
    result2 = pool2

# 第三个卷积层 size:56
# 卷积核5[3, 3, 128, 256]
# 卷积核6[3, 3, 256, 256]
# 卷积核7[3, 3, 256, 256]
with tf.name_scope('conv3_layer'):
    w_conv5 = weight_variable([3, 3, 128, 256])
    b_conv5 = bias_variable([256])
    conv_kernel5 = conv2d(result2, w_conv5)
    bn5 = tf.layers.batch_normalization(conv_kernel5, training=is_training)
    conv5 = tf.nn.relu(tf.nn.bias_add(bn5, b_conv5))

    w_conv6 = weight_variable([3, 3, 256, 256])
    b_conv6 = bias_variable([256])
    conv_kernel6 = conv2d(conv5, w_conv6)
    bn6 = tf.layers.batch_normalization(conv_kernel6, training=is_training)
    conv6 = tf.nn.relu(tf.nn.bias_add(bn6, b_conv6))

    w_conv7 = weight_variable([3, 3, 256, 256])
    b_conv7 = bias_variable([256])
    conv_kernel7 = conv2d(conv6, w_conv7)
    bn7 = tf.layers.batch_normalization(conv_kernel7, training=is_training)
    conv7 = tf.nn.relu(tf.nn.bias_add(bn7, b_conv7))

    pool3 = max_pool_2x2(conv7)  # 56*56 -> 28*28
    result3 = pool3

# 第四个卷积层 size:28
# 卷积核8[3, 3, 256, 512]
# 卷积核9[3, 3, 512, 512]
# 卷积核10[3, 3, 512, 512]
with tf.name_scope('conv4_layer'):
    w_conv8 = weight_variable([3, 3, 256, 512])
    b_conv8 = bias_variable([512])
    conv_kernel8 = conv2d(result3, w_conv8)
    bn8 = tf.layers.batch_normalization(conv_kernel8, training=is_training)
    conv8 = tf.nn.relu(tf.nn.bias_add(bn8, b_conv8))

    w_conv9 = weight_variable([3, 3, 512, 512])
    b_conv9 = bias_variable([512])
    conv_kernel9 = conv2d(conv8, w_conv9)
    bn9 = tf.layers.batch_normalization(conv_kernel9, training=is_training)
    conv9 = tf.nn.relu(tf.nn.bias_add(bn9, b_conv9))

    w_conv10 = weight_variable([3, 3, 512, 512])
    b_conv10 = bias_variable([512])
    conv_kernel10 = conv2d(conv9, w_conv10)
    bn10 = tf.layers.batch_normalization(conv_kernel10, training=is_training)
    conv10 = tf.nn.relu(tf.nn.bias_add(bn10, b_conv10))

    pool4 = max_pool_2x2(conv10)  # 28*28 -> 14*14
    result4 = pool4

# 第五个卷积层 size:14
# 卷积核11[3, 3, 512, 512]
# 卷积核12[3, 3, 512, 512]
# 卷积核13[3, 3, 512, 512]
with tf.name_scope('conv5_layer'):
    w_conv11 = weight_variable([3, 3, 512, 512])
    b_conv11 = bias_variable([512])
    conv_kernel11 = conv2d(result4, w_conv11)
    bn11 = tf.layers.batch_normalization(conv_kernel11, training=is_training)
    conv11 = tf.nn.relu(tf.nn.bias_add(bn11, b_conv11))

    w_conv12 = weight_variable([3, 3, 512, 512])
    b_conv12 = bias_variable([512])
    conv_kernel12 = conv2d(conv11, w_conv12)
    bn12 = tf.layers.batch_normalization(conv_kernel12, training=is_training)
    conv12 = tf.nn.relu(tf.nn.bias_add(bn12, b_conv12))

    w_conv13 = weight_variable([3, 3, 512, 512])
    b_conv13 = bias_variable([512])
    conv_kernel13 = conv2d(conv12, w_conv13)
    bn13 = tf.layers.batch_normalization(conv_kernel13, training=is_training)
    conv13 = tf.nn.relu(tf.nn.bias_add(bn13, b_conv13))

    pool5 = max_pool_2x2(conv13)  # 14*14 -> 7*7
    result5 = pool5

# 第一个全连接层 size:7
# 隐藏层节点数 4096
with tf.name_scope('fc1_layer'):
    w_fc14 = weight_variable([7 * 7 * 512, 4096])
    b_fc14 = bias_variable([4096])
    result5_flat = tf.reshape(result5, [-1, 7 * 7 * 512])
    fc14 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result5_flat, w_fc14), b_fc14))
    result6 = fc14
    # result6 = tf.nn.dropout(fc14, keep_prob)

# 第二个全连接层
# 隐藏层节点数 4096
with tf.name_scope('fc2_layer'):
    w_fc15 = weight_variable([4096, 4096])
    b_fc15 = bias_variable([4096])
    fc15 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result6, w_fc15), b_fc15))
    result7 = fc15
    # result7 = tf.nn.dropout(fc15, keep_prob)

# 输出层
with tf.name_scope('output_layer'):
    w_fc16 = weight_variable([4096, classify_num])
    b_fc16 = bias_variable([classify_num])
    fc16 = tf.matmul(result7, w_fc16) + b_fc16
    logits = tf.nn.softmax(fc16)

# 损失函数
with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc16, labels=y_input)
    loss = tf.reduce_mean(cross_entropy)

# 训练函数
with tf.name_scope('train'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):  # 保证train_op在update_ops执行之后再执行。
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

# 计算准确率
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# 会话初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
save_dir = "classify_modles"
checkpoint_name = "train.ckpt"
merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
writer = tf.summary.FileWriter('logs', sess.graph)  # 将训练日志写入到logs文件夹下

# 变量初始化
training_steps = 30000
display_step = 10
batch_size = 15
train_images_count = 0
test_images_count = 0

# 训练
print("Training start...")

# 模型恢复
sess = tf.InteractiveSession()
saver.restore(sess, os.path.join(save_dir, checkpoint_name))
print("Model restore success！")


for step in range(training_steps):
    train_images, train_labels, train_images_count = combine_train_dataset(train_images_count, batch_size)

    train_step.run(feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 0.5, is_training: True})
    if step % display_step == 0:
        train_accuracy = accuracy.eval(
            feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 1.0, is_training: False})
        # train_loss = sess.run(loss, feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 0.5})
        test_result = sess.run(tf.argmax(logits, 1),
                               feed_dict={x_input: train_images, keep_prob: 1.0, is_training: False})
        test_label = sess.run(tf.argmax(y_input, 1), feed_dict={y_input: train_labels})
        print(test_result)
        print(test_label)
        print("step {}\n training accuracy {}\n".format(step, train_accuracy))
        result = sess.run(merged, feed_dict={x_input: train_images, y_input: train_labels, keep_prob: 1.0,
                                             is_training: False})  # 计算需要写入的日志数据
        writer.add_summary(result, step)  # 将日志数据写入文件

    if step % (display_step*10) == 0:
        saver.save(sess, os.path.join(save_dir, checkpoint_name))
        print("Model save success!\n")

print("Training finish...")

# 模型保存
saver.save(sess, os.path.join(save_dir, checkpoint_name))
print("\nModel save success!")

print("\nTesting start...")
avg_accuracy = 0
for i in range(int(test_images_num / 30) + 1):
    test_images, test_labels, test_images_count = combine_test_dataset(test_images_count, 30)
    test_accuracy = accuracy.eval(
        feed_dict={x_input: test_images, y_input: test_labels, keep_prob: 1.0, is_training: False})
    test_result = sess.run(tf.argmax(logits, 1), feed_dict={x_input: test_images, keep_prob: 1.0, is_training: False})
    test_label = sess.run(tf.argmax(y_input, 1), feed_dict={y_input: test_labels})
    print(test_result)
    print(test_label)
    print("test accuracy {}".format(test_accuracy))
    avg_accuracy += test_accuracy

print("\ntest_avg_accuracy {}".format(avg_accuracy / (int(test_images_num / 30) + 1)))

sess.close()


# # 识别
# def trash_classify(img):
#     image = np.reshape(img,[1,224,224,3])
#     classify_result = sess.run(tf.argmax(logits, 1),feed_dict={x_input:image, keep_prob:1.0, is_training: False})
#     probability = sess.run(logits,feed_dict={x_input:image, keep_prob:1.0}).flatten().tolist()
#     return classify_result, probability
#
# test_image = cv2.imread("test.jpg")
# print(trash_classify(test_image))
