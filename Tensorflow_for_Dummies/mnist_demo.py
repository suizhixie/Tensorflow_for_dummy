from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder("float", [None, 784])
#占位符   将每一张图展平成784维的向量
w = tf.Variable(tf.zeros([784,10]))
#mnist数据集 每一张图片包含28*28的像素 把像素展开成一个向量长度是784
# mnist.train.images是一个形状为【60000,784】的张量
# 第一个维度数字来索引图片 第二个维度数字来索引像素点
b = tf.Variable(tf.zeros([10]))
#设置 b w的初值
y = tf.nn.softmax(tf.matmul(x,w) + b)
y_ = tf.placeholder("float", [None,10])
#计算交叉熵   预测分布的对数 × 真实值 的和
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#训练模型                                      学习速率 0.01
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #每次随机抓取100个批处理点   用这些批处理点替换占位符
    sess.run(train_step, feed_dict={x:batch_xs,y_:batch_ys})
#评估模型
correct_prediction = tf.equal(tf.arg_max(y,1), tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(sess.run(accuracy , feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
