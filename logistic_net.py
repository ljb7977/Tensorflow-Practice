import tensorflow as tf

x_data = [
    [0, 0, 1, 1],
    [0, 1, 0, 1]
]

y_data = [
    [0, 1, 1, 0]
]

X = tf.placeholder(tf.float32, shape=[2, None])
Y = tf.placeholder(tf.float32, shape=[1, None])
W1 = tf.constant([[5, 5], [-7, -7]], dtype=tf.float32)
b1 = tf.constant([[-8], [3]], dtype=tf.float32)
W2 = tf.constant([[-11, -11]], dtype=tf.float32)
b2 = tf.constant([6], dtype=tf.float32)

Y1 = tf.sigmoid(tf.matmul(W1, X)+b1)
Y_ = tf.sigmoid(tf.matmul(W2, Y1)+b2)

sess = tf.Session()

print(sess.run(Y_, feed_dict={X: x_data, Y: y_data}))

