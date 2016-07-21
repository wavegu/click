import numpy
import sys


class TrainingSet:
	def __init__(self, instances):
		self.instances = instances
		numInstances = len(self.instances)
		sampleWeights = numpy.zeros(numInstances, dtype = numpy.longdouble)
		for i in range(numInstances):
			sampleWeights[i] = numpy.abs(self.instances[i].loss * numpy.exp(self.instances[i].invLogPropensity))

		self.sampleWeights = sampleWeights
		# print("TrainingSet:init\t[LOG]\tNumInstances: %d |Sample weight| [min, max, mean]:" % numInstances,
		# 		self.sampleWeights.min(), self.sampleWeights.max(), self.sampleWeights.mean(), flush=True)
		self.trainIndices = None
		self.holdoutIndices = None

		self.meanConstant = None
		self.sqConstant = None
		self.cConstant = None

	def shuffle(self, mini_batch, holdout_period):
		numInstances = None
		sortIndices = None
		if holdout_period <= 0:
			if self.trainIndices is None:
				print("TrainingSet:shuffle\t[ERR]\tTrain indices were not set. Call TrainingSet:shuffle with a positive holdout_period first.", flush=True)
				sys.exit(0)

			numInstances = numpy.shape(self.trainIndices)[0]
			sortIndices = numpy.argsort(-self.sampleWeights[self.trainIndices], axis = None)
			sortIndices = self.trainIndices[sortIndices]
		else:
			numInstances = len(self.instances)
			sortIndices = numpy.argsort(-self.sampleWeights, axis = None)

		partitionSize = int(numInstances * 1.0 / mini_batch)
		perPartitionElements = []
		for i in range(mini_batch):
			currList = None
			if i < (mini_batch - 1):
				currList = sortIndices[i*partitionSize:(i+1)*partitionSize].copy()
			else:
				currList = sortIndices[i*partitionSize:].copy()

			numpy.random.shuffle(currList)
			perPartitionElements.append(currList.tolist())

		shuffledOrder = []
		currIndex = 0
		while len(shuffledOrder) < numInstances:
			currList = perPartitionElements[currIndex]
			currIndex += 1
			if currIndex >= mini_batch:
				currIndex = 0
			if len(currList) <= 0:
				continue
			chosenElement = currList.pop()
			shuffledOrder.append(chosenElement)

		if holdout_period <= 0:
			self.trainIndices = numpy.array(shuffledOrder)
			return shuffledOrder

		#Divide into train and hold-out sets otherwise
		trainFraction = []
		holdoutFraction = []
		for i in range(partitionSize):
			if i < (partitionSize - 1):
				currList = shuffledOrder[i*mini_batch : (i+1)*mini_batch]
			else:
				currList = shuffledOrder[i*mini_batch : ]

			if i % holdout_period == 0:
				holdoutFraction.extend(currList)
			else:
				trainFraction.extend(currList)

		self.holdoutIndices = numpy.array(holdoutFraction)
		self.trainIndices = numpy.array(trainFraction)
		return trainFraction, holdoutFraction

	def holdout_score(self, weights):
		if self.holdoutIndices is None:
			print("TrainingSet:holdout_score\t[ERR]\tHoldout indices were not set. Call TrainingSet:shuffle with a positive holdout_period first.", flush=True)
			sys.exit(0)

		estimatedRisk = 0.0
		for ind in self.holdoutIndices:
			instance = self.instances[ind]
			risk, grad = instance.risk_gradient(weights, -1, False)
			estimatedRisk += risk

		estimatedRisk = estimatedRisk * 1.0 / numpy.shape(self.holdoutIndices)[0]
		return estimatedRisk

	def compute_constants(self, weights, clip_value, var_penalty):
		if var_penalty <= 0:
			self.meanConstant = 1.0
			self.sqConstant = 0.0
			return

		if self.trainIndices is None:
			print("TrainingSet:compute_constants\t[ERR]\Train indices were not set. Call TrainingSet:shuffle with a positive holdout_period first.", flush=True)
			sys.exit(0)

		numTrainIndices = numpy.shape(self.trainIndices)[0]
		estimatedRisks = numpy.zeros(numTrainIndices, dtype = numpy.longdouble)
		for i in range(numTrainIndices):
			instance = self.instances[self.trainIndices[i]]
			risk, grad = instance.risk_gradient(weights, clip_value, False)
			estimatedRisks[i] = risk

		stdRisk = estimatedRisks.std(dtype = numpy.longdouble, ddof = 1)
		meanRisk = estimatedRisks.mean(dtype = numpy.longdouble)

		self.meanConstant = 1 - var_penalty * numpy.sqrt(numTrainIndices) * meanRisk / ((numTrainIndices - 1)*stdRisk)
		self.sqConstant = var_penalty * numpy.sqrt(numTrainIndices) / (2 * (numTrainIndices - 1) * stdRisk)

	def update(self, weights, batch_id, batch_size, clip_value, l2_penalty, adagrad_divider):
		numTrainInstances = numpy.shape(self.trainIndices)[0]
		numBatches = int(numTrainInstances * 1.0 / batch_size)
		currIndices = None
		if batch_id < (numBatches - 1):
			currIndices = self.trainIndices[batch_id*batch_size: (batch_id+1)*batch_size]
		else:
			currIndices = self.trainIndices[batch_id*batch_size:]

		gradient = None
		estimatedRisk = 0.0
		for ind in currIndices:
			instance = self.instances[ind]
			risk, grad = instance.risk_gradient(weights, clip_value, True)
			estimatedRisk += risk
			if gradient is None:
				gradient = grad
			else:
				gradient += grad

		estimatedRisk = estimatedRisk / numpy.shape(currIndices)[0]
		gradient = gradient / numpy.shape(currIndices)[0]
		gradient = numpy.divide(gradient, adagrad_divider)

		adagrad_divider = numpy.sqrt(numpy.square(adagrad_divider) + numpy.square(gradient))

		updateDirection = l2_penalty * weights + (self.meanConstant + self.sqConstant * 2 * estimatedRisk) * gradient
		return weights - 0.5 * updateDirection, adagrad_divider
