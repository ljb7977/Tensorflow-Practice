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

layer1 = tf.layers.dense(X, 2, activation=tf.nn.sigmoid)
layer2 = tf.layers.dense(layer1, 1, activation=None)

# -----version 1--------
# Y_ = tf.nn.sigmoid(layer2)
# cost = -tf.reduce_mean(Y*tf.log(Y_)+(1-Y)*tf.log(1-Y_))

# -----version 2-----
Y_ = layer2
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_, labels=Y))


train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.round(tf.nn.sigmoid(Y_))
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(100000):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)
        if cost_val < 0.02:
            break

    h, p, a = sess.run([Y_, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print("Hypothesis: ", h, "\nPredicted: ", p, "\nAccuracy: ", a)
