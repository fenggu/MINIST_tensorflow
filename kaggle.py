import pandas as pd
import os
import minist
import tensorflow as tf
train = minist.train
test = minist.test
validation = minist.validation
sess = tf.InteractiveSession()
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

"""
    测试数据时feed全部数据可能会造成Out of memory 异常，这里将test集拆解为mini-batch
"""
def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #28*28*32
h_pool1 = max_pool_2x2(h_conv1) #14*14*32

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #14*14*64
h_pool2 = max_pool_2x2(h_conv2) #7*7*64

# 全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# 训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 训练步数记录
global_step = tf.Variable(0, name='global_step', trainable=False)

# 存档入口
saver = tf.train.Saver()
# 在Saver声明之后定义的变量将不会被存储

ckpt_dir = './kaggle_ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

sess.run(tf.global_variables_initializer())
# 加载模型存档
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
if ckpt and ckpt.model_checkpoint_path:
    print('Restoring from checkpoint: %s' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

start = global_step.eval()
for i in range(start, start + 1):
    batch = train.next_batch(50)
    if i%200 == 0:
        train_accuracy = accuracy.eval(feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        # 模型存档
        global_step.assign(i).eval()
        saver.save(sess, ckpt_dir + '/logistic.ckpt', global_step=i)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={ x: validation.images, y_: validation.labels, keep_prob: 1.0}))

batchs = get_batchs(test, 50)
labels = []
imageId = []
id = 1
for test_image in batchs:
    prediction = tf.argmax(y_conv, 1)
    test_labels = prediction.eval(feed_dict={x: test_image, keep_prob: 1.0})
    for label in test_labels:
        labels.append(label)
        imageId.append(id)
        id += 1

submission = pd.DataFrame({
    "ImageId": imageId,
    "Label": labels
})
submission.to_csv("data/minist_submission.csv", index=False)
print("submission ok")

