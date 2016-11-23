from collections import namedtuple
import numpy
import random



Datapoint = namedtuple('Datapoint', ['features', 'goldlabel'])

class IrisDataReader():

	def __init__(self,filename):
		self.filename = filename
		self._read()

	def _read(self):
		self.data = []
		with open(self.filename) as f:
			for line in f:

				line = line.strip()
				line = line.split(" ")
				features = [float(dim.split(":")[1]) for dim in line[1:]]

				self.data.append(Datapoint(numpy.array(features),numpy.float32(line[0])))

	def __iter__(self):
		for datapoint in self.data:
			yield datapoint

	def __len__(self):
		return len(self.data)

	def __getitem__(self,key):
		return self.data[key]



	def feature_dimensions(self):
		for datapoint in self:
			 return len(datapoint.features)

def constantlearningrate():
	while True:
		yield 1

def desclearningrate(n=1):
	i = 1
	while True:
		yield n/i

class Perceptron():

	def __init__(self,dimensions,*,init_weights = "zeros",init_bias = numpy.float32(1),learningrate = desclearningrate()):
		
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

		bias_updates = [0.5 * (datapoint.goldlabel - numpy.sign(self.predict(datapoint))) * learningrate for datapoint in datapoints]
		updates = [0.5 * (datapoint.goldlabel - numpy.sign(self.predict(datapoint))) * datapoint.features * learningrate for datapoint in datapoints]

		self.bias += sum(bias_updates) * 1/len(datapoints)
		self.weights += sum(updates) * 1/len(datapoints)

Results = namedtuple('Results', ['all', 'correct', 'misclassified'])

class Experiment():

	def __init__(self,trainset,testset,*,batchsize = 1,learningrate = desclearningrate(),epochs = 1):

		assert trainset.feature_dimensions() == testset.feature_dimensions()

		self.trainset, self.testset = trainset,testset
		self.batchsize, self.epochs = min(len(trainset),batchsize), epochs
		self.perceptron = Perceptron(trainset.feature_dimensions(),learningrate = learningrate)

	def run(self):
		self._train(self.trainset,batchsize = self.batchsize,epochs=self.epochs)
		return self._evaluate(self.testset)

	def _train(self,trainset,*,epochs = 1, batchsize = 1):

		datacount = len(trainset)
		for _ in range(epochs):
			assert batchsize <= datacount
			for _ in range(datacount//batchsize):
				batch = random.sample(list(trainset),batchsize)
				self.perceptron.update(batch)

	def _evaluate(self,testset):

		correctness = [self.perceptron.predict(datapoint) * datapoint.goldlabel for datapoint in testset]

		count_correct = 0
		count_false = 0
		for i in correctness:
			if i > 0:
				count_correct += 1
			else:
				count_false += 1

		return Results(count_correct+count_false,count_correct,count_false)

if __name__ == "__main__":

	trainset = IrisDataReader("data/iris.setosa-v-rest.train")
	testset = IrisDataReader("data/iris.setosa-v-rest.test")

	e1 = Experiment(trainset,testset,epochs = 1, batchsize = 10, learningrate = constantlearningrate())
	print(e1.run())

	e2 = Experiment(trainset,testset,epochs = 1, batchsize = 1, learningrate = desclearningrate())
	print(e2.run())
	
	
