import numpy
import Dataset
import sklearn.linear_model

class LinearRegression:
	def __init__(self):
		self.classifier = sklearn.linear_model.LinearRegression()

	def get_X_losses(self, instances):
		X = []
		losses = []
		for ins in instances:
			chosenFeature = ins.x[:, ins.y].tolist()
			X.append(chosenFeature)
			losses.append(ins.loss)
		X = numpy.asarray(X)
		losses = numpy.asarray(losses)
		return X, losses

	def train(self, trainInstances):
		X, losses = self.get_X_losses(trainInstances)
		self.classifier.fit(X, losses)

	def predict(self, testInstances):
		X, losses = self.get_X_losses(testInstances)
		prediction = self.classifier.predict(X)
		print(prediction)


if __name__ == '__main__':
	dataset = Dataset.Dataset()
	instances = dataset.read_bandit_data('../data/Brute.txt')
	regressor = LinearRegression()
	regressor.train(instances)
	regressor.predict(instances)
