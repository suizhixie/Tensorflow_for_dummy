from tensorflow.examples.tutorials.mnist import input_data
import  tensorflow as tf





mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
#占位符  方便运行时将数据填充
x = tf.placeholder("float", shape=[None,784])

y_ = tf.placeholder("float", shape=[None,784])

w = tf.Variable(tf.zeros([784,10]))
#表示w有784个特征值 和10个输出值

b = tf.Variable(tf.zeros([10]))
# b是一个十维向量 因为我们有十个分类
sess.run(tf.initialize_all_variables())

#类别预测与损失函数
y = tf.nn.softmax(tf.matmul(x,w) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    #每一步迭代加载50个样本
    train_step.run(feed_dict={x: batch[0],y_:batch[1]})

correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
#tf.argmax(）返回在某一维上其数据的最大值所在索引，由于标签向量是由0,1组成
#所以最大值1所在的索引位置就是类别标签   tf.arg_max(y,1)就代表预测的标签

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print(accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels}))


#构建一个多层卷积网络

#权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return  tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


#卷积和池化
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1]
                          ,padding='SAME')

#第一层卷积
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable(([32]))

x_image = tf.reshape(x, [-1,28,28,1])
h_cov1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_cov1)


#第二层卷积
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#dropout  减少过拟合
keep_prob = tf.placeholder("float")
h_fc_1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc2) + b_fc2)

#模型评估
cross_entropy  = -tf.reduce_sum(y_*log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0],y_:batch[1],keep_prob:1.0
        })
        print("step %d training accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1], keep_prob:0.5})
print("test accuracy %g"%accuracy.eval(feed_dict = {
    x:mnist.test.images, y_:mnist.test.labels,keep_prob:1.0
}))







































