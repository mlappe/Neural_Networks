import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


def get_linear_activation():
	
	def linear_activation(values):

		zeros = tf.zeros([10])
		return tf.nn.bias_add(values, zeros, data_format=None, name=None)

	return linear_activation



class Perceptron():

	def __init__(self,input_dim, output_dim,*,activation_function = get_linear_activation()):
		"""
		for example 
				activation_function = tf.sigmoid
		"""

		self.weights = tf.Variable(tf.zeros([input_dim, output_dim]))
		self.bias = tf.Variable(tf.random_normal([output_dim]))

		self.activation_function = activation_function



	def output(self,data):

		return self.activation_function(tf.matmul(data, self.weights) + self.bias)

class Layer():
	def __init__(self,cell_factory, number_of_cells):
		self.cells = [cell_factory() for i in range(number_of_cells)]

	def output(self,data):
		outputs = [cell.output(data) for cell in self.cells]
		return tf.concat(1,outputs)


if __name__ == "__main__":
	x = tf.placeholder(tf.float32, [None, 784])
	layer1 = Layer(lambda : Perceptron(784,10,activation_function = tf.sigmoid),20)
	layer2 = Layer(lambda : Perceptron(200,10),1)
	y = tf.nn.softmax(layer2.output(layer1.output(x)))

	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	init = tf.initialize_all_variables()

	with tf.Session() as sess:

		sess.run(init)


		for i in range(1000):
			batch_xs, batch_ys = mnist.train.next_batch(100)
			sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
