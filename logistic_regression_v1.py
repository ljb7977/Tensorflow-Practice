import tensorflow as tf

x_data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y_data = [
    [0],
    [1],
    [1],
    [0]
]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([1, 2]))
b = tf.Variable(tf.random_normal([1]))

Y_ = tf.sigmoid(tf.matmul(W, tf.transpose(X))+b)

print(Y_)
print(Y)
cost = -tf.reduce_mean(tf.transpose(Y)*tf.log(Y_)+(1-tf.transpose(Y))*tf.log(1-Y_))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(Y_ > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(100000):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, p, a = sess.run([Y_, predicted, accuracy], feed_dict={X: x_data, Y:y_data})

    print("Hypothesis: ", h, "\nPredict: ", p, "\nAccuracy: ", a)

