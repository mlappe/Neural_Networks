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
				yield Datapoint(numpy.array(features),line[0])

class Perceptron():
	pass

if __name__ == "__main__":
	dataset = DataReader("data/iris.setosa-v-rest.train")
	for line in dataset:
		print(line)

	print(list(dataset))
	
