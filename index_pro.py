import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import os
sess = tf.InteractiveSession()
# 为了创建这个模型，我们需要创建大量的权重和偏置项。
# 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
# 由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。
# 为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。
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
global_step = tf.Variable(0, name='global_step', trainable=False)


# 训练步数记录
# 存档入口
saver = tf.train.Saver()

# 在Saver声明之后定义的变量将不会被存储
# non_storable_variable = tf.Variable(777)

ckpt_dir = './ckpt_dir'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

sess.run(tf.global_variables_initializer())
# 加载模型存档
ckpt = tf.train.get_checkpoint_state(ckpt_dir)
if ckpt and ckpt.model_checkpoint_path:
    print('Restoring from checkpoint: %s' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

start = global_step.eval()
for i in range(start, start + 10000):
  batch = mnist.train.next_batch(50)
  if i%10 == 0:
    train_accuracy = accuracy.eval(feed_dict={
      x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    # 模型存档
  global_step.assign(i).eval()
  saver.save(sess, ckpt_dir + '/logistic.ckpt', global_step=i)
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

