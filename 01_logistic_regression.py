import tensorflow as tf

x_data = [
    [1, 2, 3, 4, 5, 6],
    [2, 3, 1, 3, 3, 2]
]

y_data = [
    [0, 0, 0, 1, 1, 1]
]

X = tf.placeholder(tf.float32, shape=[2, None])
Y = tf.placeholder(tf.float32, shape=[1, None])
W = tf.Variable(tf.random_normal([1, 2]))
b = tf.Variable(tf.random_normal([1]))

Y_ = tf.sigmoid(tf.matmul(W, X)+b)

cost = -tf.reduce_mean(Y*tf.log(Y_)+(1-Y)*tf.log(1-Y_))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(Y_ > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

print(predicted)
print(accuracy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(50000):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, p, a = sess.run([Y_, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print("Hypothesis: ", h, "\nPredicted: ", p, "\nAccuracy: ", a)
