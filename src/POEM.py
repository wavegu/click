import numpy
import Dataset
import Instance
import TrainingSet


class POEM:
	def __init__(self):
		self.instances = None
		self.bestWeights = None

	def load_data(self, filePath):
		dataset = Dataset.Dataset()
		self.instances = dataset.read_bandit_data(filePath, 'Brute')

	def train(self, trainInstances, args):
		# init
		weights = trainInstances[0].parametrize()
		numInstances = len(trainInstances)
		losses = numpy.zeros(numInstances, dtype = numpy.longdouble)
		for i in range(numInstances):
			losses[i] = trainInstances[i].loss

		translation = numpy.percentile(losses, args.norm*100, axis = None)
		# print("POEM_learn:main\t[LOG]\t[Min,Max,Mean] Loss: ", losses.min(), losses.max(), losses.mean(), flush=True)
		# print("POEM_learn:main\t[LOG]\tSelf-normalization fraction and chosen translation: ",
				# args.norm, translation, flush=True)
		for i in range(numInstances):
			trainInstances[i].loss = trainInstances[i].loss - translation
		clipValue = -1
		if args.clip >= 0:
			propensities = numpy.zeros(numInstances, dtype = numpy.longdouble)
			for i in range(numInstances):
				propensities[i] = numpy.exp(trainInstances[i].invLogPropensity)

			clipValue = numpy.percentile(propensities, args.clip*100, axis = None)
			# print("POEM_learn:main\t[LOG]\t[Min,Max,Mean] InvPropensity: ", propensities.min(), propensities.max(), propensities.mean(), flush=True)
			# print("POEM_learn:main\t[LOG]\tClip percentile and chosen clip constant: ", args.clip, clipValue, flush=True)

		trainSet = TrainingSet.TrainingSet(trainInstances)
		trainSet.shuffle(args.minibatch, args.progvalidate)

		bestWeights = None
		bestScore = None
		epochID = 0
		adagradDecay = numpy.ones(weights.shape, dtype = numpy.longdouble)
		patience = 0
		while True:
			epochID += 1
			# print("POEM_learn:main\t[LOG]\tStarting epoch: ", epochID, flush=True)
			#At the start of an epoch, shuffle training set
			trainSet.shuffle(args.minibatch, -1)
			#Also, update the majorization constants
			trainSet.compute_constants(weights, clipValue, args.var)

			#Also, compute holdout score for the current weight vector and update best candidate so far
			score = trainSet.holdout_score(weights)
			# print("POEM_learn:main\t[LOG]\tEPOCH Current score: %0.3f Best score: " % score, bestScore, flush=True)
			if (bestScore is None) or (score < bestScore):
				bestWeights = weights.copy()
				bestScore = score
				patience = 0
			else:
				if epochID > 3:
					patience += 1

			if patience > 5:
				break

			#Process batches in this epoch
			numTrainInstances = numpy.shape(trainSet.trainIndices)[0]
			numBatches = int(numTrainInstances * 1.0 / args.minibatch)
			for i in range(numBatches):
				weights, adagradDecay = trainSet.update(weights, i, args.minibatch, clipValue, args.l2, adagradDecay)

				#If we have processed holdout_period number of batches, time to snapshot weights and update constants
				if (i+1) % args.progvalidate == 0:
					score = trainSet.holdout_score(weights)
					# print("POEM_learn:main\t[LOG]\tBATCH Current score: %0.3f Best score: " % score, bestScore, flush=True)
					if (bestScore is None) or (score < bestScore):
						bestWeights = weights.copy()
						bestScore = score
						patience = 0
					else:
						if epochID > 3:
							patience += 1

					trainSet.compute_constants(weights, clipValue, args.var)

				if patience > 5:
					break

			if patience > 5:
				break

		self.bestWeights = bestWeights

	def predict(self, testInstances):
	    estimatedRisk = 0.0
	    for instance in testInstances:
	        risk, grad = instance.risk_gradient(self.bestWeights, -1, False)
	        estimatedRisk += risk

	    estimatedRisk /= len(testInstances)
	    print("POEM_predict:main\t[LOG]\tPerformance: ", estimatedRisk, flush=True)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='POEM: Policy Optimizer for Exponential Models.')
	parser.add_argument('--clip', '-c', metavar='C', type=float,
						help='Clipping hyper-parameter', default=-1)
	parser.add_argument('--l2', '-l', metavar='L', type=float,
						help='L2 regularization hyper-parameter', default=0.0)
	parser.add_argument('--var', '-v', metavar='V', type=float,
						help='Variance regularization hyper-parameter', default=0.0)
	parser.add_argument('--norm', '-n', metavar='N', type=float,
						help='Self-normalization hyper-parameter', default=1.0)
	parser.add_argument('--minibatch', '-m', metavar='M', type=int,
						help='Minibatch size', default=10)
	parser.add_argument('--progvalidate', '-p', metavar='P', type=int,
						help='Holdout period for progressive validation', default=10)
	parser.add_argument('--seed', '-s', metavar='S', type=int,
						help='Random number seed', default=387)

	args = parser.parse_args()

	poem = POEM()
	poem.load_data('../data/Brute.txt')
	poem.train(poem.instances, args)
	poem.predict(poem.instances)
