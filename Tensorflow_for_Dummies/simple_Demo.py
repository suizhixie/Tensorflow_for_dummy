import tensorflow as tf
import numpy as np

X_data = np.float32(np.random.rand(2,100))
#返回一个均匀分布在【0,1）的随机样本值 2行100列
y_data = np.dot([0.100,0.200], X_data) + 0.300
#矩阵乘法 两列*2行 最后得到一行


#构造一个线性模型
# Variable定义图变量
b = tf.Variable(tf.zeros([1]))
#一维数组中放一个值0 即【0】
w = tf.Variable(tf.random_uniform([1], -1.0,1.0))
#生成1维的均匀分布的随机数 取值-1到1之间
y = tf.matmul(w,X_data) + b

#最小化方差  以估计值与实际值的均方误差作为损失
loss = tf.reduce_mean(tf.square(y - y_data))
#采用梯度下降法来优化参数--优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# 最小化均方误差
#初始化变量
init = tf.initialize_all_variables()

#启动图
sess = tf.Session()
sess.run(init)
#拟合平面
for step in range(0,201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(w),sess.run(b))





