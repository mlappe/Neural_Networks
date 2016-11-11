from collections import namedtuple
import numpy



Datapoint = namedtuple('Datapoint', ['features', 'goldlabel'])

class IrisDataReader():

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

	def feature_dimensions(self):
		for datapoint in self:
			 return len(datapoint.features)

def constantlearningrate():
	while True:
		yield 1

class Perceptron():
	def __init__(self,dimensions,*,init_weights = "zeros",init_bias = numpy.float32(1),learningrate = constantlearningrate()):
		

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

		bias_updates = [0.5 * (datapoint.goldlabel - self.predict(datapoint)) * learningrate for datapoint in datapoints]
		updates = [0.5 * (datapoint.goldlabel - numpy.sign(self.predict(datapoint))) * datapoint.features * learningrate for datapoint in datapoints]

		self.bias += sum(bias_updates) * 1/len(datapoints)
		self.weights += sum(updates) * 1/len(datapoints)

class Experiment():
	def __init__(self,trainset,testset,*,batchsize = 1):

		assert trainset.feature_dimensions() == testset.feature_dimensions()

		self.perceptron = Perceptron(trainset.feature_dimensions())

if __name__ == "__main__":

	trainset = IrisDataReader("data/iris.setosa-v-rest.train")
	testset = IrisDataReader("data/iris.setosa-v-rest.test")

	e = Experiment(trainset,testset)
	
	
