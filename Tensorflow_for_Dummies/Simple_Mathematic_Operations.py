import tensorflow as tf

#math with Constant Tensors
const_a = tf.constant(3,6)
const_b = tf.constant(1,2)
total = const_a + const_b
quot = tf.div(const_a,const_b)


#Math with Random Tensors
rand_a = tf.random_normal([3], 2.0)
rand_b = tf.random_normal([3],1.0 , 4.0)
diff = tf.subtract(rand_a, rand_b)


#vector multiplication
vec_a = tf.linspace(0.0 , 3.0, 4)
vec_b = tf.fill([4,1], 2.0)
prob = tf.multiply(vec_a , vec_b)
dot = tf.tensordot(vec_a, vec_b, 1)






















