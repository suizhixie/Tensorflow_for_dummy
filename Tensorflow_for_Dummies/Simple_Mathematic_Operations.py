import tensorflow as tf

#math with Constant Tensors
const_a = tf.constant(3.6)
const_b = tf.constant(1.2)
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

#Matrix Multiplication
mat_a = tf.constant([[2,3],[1,2],[4,5]])
mat_b = tf.constant([[6,4,4],[3,7,2]])

mat_prod = tf.matmul(mat_a,mat_b)

#execute the operation
with tf.Session() as sess:
    print("Sum:%f"% sess.run(total))
    print("Quotient:%f"%sess.run(quot))
    #求商
    print("Difference:",sess.run(diff))
    print("Element_wise product:",sess.run(prob))
    print("Dot product:",sess.run(mat_prod))
    print("Matrix product:",sess.run(mat_prod))



















