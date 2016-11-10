from collections import namedtuple
import numpy



Datapoint = namedtuple('Datapoint', ['features', 'goldlabel'])

class DataReader():

	def __init__(self,filename):
		self.filename = filename

	def __iter__(self):
		"""
		iterates over all datapoints in the file
		yields Datapoint objects, not the string
		"""
		with open(self.filename) as f:
			for line in f:

				line = line.strip()
				line = line.split(" ")
				features = [float(dim.split(":")[1]) for dim in line[1:]]

				yield Datapoint(numpy.array(features),numpy.float32(line[0]))

def constantlearningrate():
	while True:
		yield 1

class Perceptron():
	def __init__(self,dimensions,*,init_weights = "zeros",init_bias = numpy.float32(0.0),learningrate = constantlearningrate()):
		

		if init_weights == "zeros":
			self.weights = numpy.zeros(dimensions)
		else:
			self.weights = init_weights
		assert type(self.weights) is numpy.ndarray

		assert type(init_bias) is numpy.float32
		self.bias  = init_bias

		self.learningrate = learningrate

	def predict(self,datapoint):
		return numpy.dot(datapoint.features,self.weights) + self.bias

	def update(self,datapoints):
		assert type(datapoints) is list
		learningrate = next(self.learningrate)

		bias_updates = [(datapoint.goldlabel - self.predict(datapoint)) * datapoint.bias * learningrate for datapoint in datapoints]
		updates = [(datapoint.goldlabel - self.predict(datapoint)) * datapoint.features * learningrate for datapoint in datapoints]

		self.bias += sum(bias_updates)
		self.weights += sum(updates)

if __name__ == "__main__":

	dataset = DataReader("data/iris.setosa-v-rest.train")
	print(Perceptron(5).bias)
	p = Perceptron(4)
	for line in dataset:
		print(line)
		p.update([line])
	print(p.weights)

	list(dataset)
	
	
