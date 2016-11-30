import tensorflow as tf
import collections

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

def constantlearningrate(n=1):
	while True:
		yield n

def desclearningrate(n=1):
	i = 1
	while True:
		yield n/i

Result = collections.namedtuple("Result",["Iteration","Testaccuracy","Trainaccuracy"])

class Experiment():
	def __init__(self,*,layer1_outputsize = 10,layer1_size = 20,iterations = 1000,learning_rate = constantlearningrate(n=0.5)):
		self.layer1_outputsize = layer1_outputsize
		self.layer1_size = layer1_size
		self.iterations = iterations
		self.learning_rate = learning_rate

	def run_stepwise(self):

		x = tf.placeholder(tf.float32, [None, 784])
		layer1 = Layer(lambda : Perceptron(784,self.layer1_outputsize,activation_function = tf.sigmoid),self.layer1_size)
		layer2 = Layer(lambda : Perceptron(self.layer1_outputsize * self.layer1_size,10),1)
		y = tf.nn.softmax(layer2.output(layer1.output(x)))

		learning_rate = tf.placeholder(tf.float32, shape=[])

		y_ = tf.placeholder(tf.float32, [None, 10])

		cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

		train_step = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

		init = tf.initialize_all_variables()



		with tf.Session() as sess:

			sess.run(init)


			for i in range(self.iterations):
				batch_xs, batch_ys = mnist.train.next_batch(100)
				sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: next(self.learning_rate)})

				correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

				testaccuracy =  sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
				trainaccuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})

				yield Result(i,testaccuracy,trainaccuracy)

	def run(self):
		return list(self.run_stepwise())

def create_image(name,results):
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as pyplot

	x,testaccuracy,trainaccuracy = zip(*results)
	
	testcurve = pyplot.plot(x,testaccuracy,label="testaccuracy")
	traincurve = pyplot.plot(x,trainaccuracy,label="trainaccuracy")
	pyplot.legend(loc="upper left")

	pyplot.savefig(name)



if __name__ == "__main__":

	print("Perceptron with 10 cells in hidden layer")
	results = []
	for result in Experiment(iterations = 100,layer1_size = 10).run_stepwise():
		print(result)
		results.append(result)

	create_image("hiddenlayer10.png",results)

	print("Perceptron with 40 cells in hidden layer")

	results = []
	for result in Experiment(iterations = 100,layer1_size = 40).run_stepwise():
		print(result)
		results.append(result)

	create_image("hiddenlayer40.png",results)
	
	print("Perceptron with 20 cells in hidden layer")
	results = []
	for result in Experiment(iterations = 100,layer1_size = 20).run_stepwise():
		print(result)
		results.append(result)

	create_image("hiddenlayer20.png.png",results)





	
	 
	

		


