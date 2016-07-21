import arff
import numpy
import os
import scipy.sparse
import sklearn.preprocessing
import Instance


class Dataset:
	def __init__(self):
		self.features = None
		self.labels = None

	def create_synthetic_data(self, num_records, num_features, num_labels):
		if num_records <= 0 or num_features <= 0 or num_labels <= 0:
			print("Dataset:create_synthetic_data\t[ERR]\tCannot create synthetic data with numRecords/numFeatures/numLabels ",
					num_records, num_features, num_labels, flush=True)
			sys.exit(0)

		features = numpy.random.randn(num_records, num_features)
		features = sklearn.preprocessing.robust_scale(features)
		self.features = sklearn.preprocessing.normalize(features)

		labels = numpy.random.random_integers(0, 1, size=(num_records, num_labels))
		self.labels = labels
		# print("Dataset:create_synthetic_data\t[LOG]\tCreated synthetic data with numRecords/numFeatures/numLabels ", num_records, num_features, num_labels, flush=True)

	def write_bandit_data(self, repo_dir, instance_type, seed):
		fileName = repo_dir + instance_type+'.txt'
		if not os.path.exists(repo_dir):
			print("Dataset:write_bandit_data\t[ERR]\tOutput directory not found at ", repo_dir, flush=True)
			sys.exit(0)

		numpy.random.seed(seed)

		numRecords, numFeatures = numpy.shape(self.features)
		numLabels = numpy.shape(self.labels)[1]

		f = open(fileName, 'w')
		#Write header
		if instance_type != 'Brute':
			f.write(str(numRecords)+' '+str(numFeatures)+' '+str(numLabels)+'\t# HEADER: numRecords numFeatures numLabels\n')
		else:
			f.write(str(numRecords)+' '+str(numLabels * numFeatures)+' '+str(numLabels)+'\t# HEADER: numRecords numFeatures maxPossibleActions\n')
		for i in range(numRecords):
			featureStr = ''
			for j in range(numFeatures):
				if self.features[i,j] != 0:
					featureStr += str(j) + ':' + str(self.features[i,j]) + ' '

			featureStr += '\t# ID:'+str(i)+'\n'

			instanceStr = ''
			if instance_type == 'MultiClass':
				sampledClass = numpy.random.randint(numLabels)
				sampledLoss = 1.0 - self.labels[i, sampledClass]
				propensity = 1.0 / numLabels
				instanceStr = str(sampledClass) + ' ' + str(sampledLoss) + ' ' + str(propensity) + ' '

			elif instance_type == 'MultiLabel':
				sampledLabels = numpy.random.random_integers(0, high=1, size = numLabels)
				sampledLoss = (self.labels[i, :] != sampledLabels).sum(dtype = numpy.longdouble)
				propensity = 1.0 / (2 ** numLabels)
				labelStr = ''
				for k in range(numLabels):
					if sampledLabels[k] > 0:
						labelStr += str(k)+','

				if labelStr == '':
					labelStr = '-1'
				else:
					labelStr = labelStr[:-1]
				instanceStr = labelStr + ' ' + str(sampledLoss) + ' ' + str(propensity) + ' '

			elif instance_type == 'Brute':
				numAllowedActions = numpy.random.random_integers(3, high=numLabels)
				allowedActions = numpy.random.choice(numLabels, size = numAllowedActions, replace = False)

				sampledClassIndex = numpy.random.randint(numAllowedActions)
				sampledClass = allowedActions[sampledClassIndex]
				sampledLoss = 1.0 - self.labels[i, sampledClass]
				propensity = 1.0 / numAllowedActions
				instanceStr = str(sampledClassIndex) + ' ' + str(sampledLoss) + ' ' + str(propensity) + ' '

				featureStr = str(numAllowedActions) + '\t# ID:'+str(i)+' -- '+ numpy.array_str(allowedActions) +'\n'
				for k in range(numAllowedActions):
					featureStr += str(k) + ' '
					currentAction = allowedActions[k]
					for j in range(numFeatures):
						if self.features[i,j] != 0:
							featureStr += str(currentAction * numFeatures + j) + ':' + str(self.features[i,j]) + ' '
					featureStr += '\t# Action: ' + str(currentAction) + '\n'

			f.write(instanceStr + featureStr)

		f.close()
		print("Dataset:write_bandit_data\t[LOG]\tOutput bandit data to ", fileName, flush=True)

	def read_bandit_data(self, repo_dir, instance_type='Brute'):
		if repo_dir.endswith('.txt'):
			fileName = repo_dir
		else:
			fileName = repo_dir + instance_type+'.txt'

		if not os.path.exists(fileName):
			print("Dataset:read_bandit_data\t[ERR]\tInput file not found at ", fileName, flush=True)
			sys.exit(0)

		f = open(fileName, 'r')
		allLines = f.readlines()
		f.close()

		# parse header
		header = allLines[0]
		commentIndex = header.find('#')
		if commentIndex >= 0:
			header = header[:commentIndex]
		tokens = header.split()
		numRecords = int(tokens[0])
		numFeatures = int(tokens[1])
		numLabels = int(tokens[2])

		currIndex = 0
		instanceList = []
		# print("Dataset:read_bandit_data\t[LOG]\tFilename: %s Number of instances: %d" %\
				# (fileName, numRecords), flush=True)

		for i in range(numRecords):
			currIndex += 1
			currentLine = allLines[currIndex]
			commentIndex = currentLine.find('#')
			if commentIndex >= 0:
				currentLine = currentLine[:commentIndex]
			tokens = currentLine.split()
			sampledAction = tokens[0]
			sampledLoss = float(tokens[1])
			sampledPropensity = float(tokens[2])

			sampledY = None
			newInstance = None
			instanceFeature = None
			if instance_type == 'MultiClass':
				newInstance = Instance.MultiClass(numLabels, numFeatures)
				sampledY = int(sampledAction)
				instanceFeature = numpy.zeros(numFeatures, dtype = numpy.longdouble)
			elif instance_type == 'MultiLabel':
				newInstance = Instance.MultiLabel(numLabels, numFeatures)
				sampledY = numpy.zeros(numLabels, dtype = numpy.int)
				if sampledAction != '-1':
					for eachLabel in sampledAction.split(','):
						sampledY[int(eachLabel)] = 1
				instanceFeature = numpy.zeros(numFeatures, dtype = numpy.longdouble)
			elif instance_type == 'Brute':
				newInstance = Instance.Brute(numFeatures)
				sampledY = int(sampledAction)

			if instance_type == 'MultiClass' or instance_type == 'MultiLabel':
				for j in range(3, len(tokens)):
					idVal = tokens[j].split(':')
					instanceFeature[int(idVal[0])] = float(idVal[1])

			elif instance_type == 'Brute':
				numActions = int(tokens[3])
				instanceFeature = numpy.zeros((numFeatures,numActions), dtype = numpy.longdouble)
				for k in range(numActions):
					currIndex += 1
					currentAction = allLines[currIndex]
					commentIndex = currentAction.find('#')
					if commentIndex >= 0:
						currentAction = currentAction[:commentIndex]
					tokens = currentAction.split()
					currentCol = int(tokens[0])
					for j in range(1, len(tokens)):
						idVal = tokens[j].split(':')
						instanceFeature[int(idVal[0]), currentCol] = float(idVal[1])

			newInstance.set(sampledPropensity, sampledLoss, instanceFeature, sampledY)
			instanceList.append(newInstance)
		# 	if i % 20 == 0:
		# 		print(".", flush=True, end='')
		# print('')
		# print("Dataset:read_bandit_data\t[LOG]\tFinished loading filename: %s Number of instances: %d" %\
		# 		(fileName, numRecords), flush=True)
		return instanceList



if __name__ == "__main__":
	seed = 387

	d1 = Dataset()
	d1.create_synthetic_data(50, 4, 3)
	d1.write_bandit_data('./', 'MultiClass', seed)
	d1.write_bandit_data('./', 'MultiLabel', seed)
	d1.write_bandit_data('./', 'Brute', seed)

	d2 = Dataset()
	d2.read_supervised_data('./')
	d2.write_bandit_data('./', 'MultiClass', seed)
	d2.write_bandit_data('./', 'MultiLabel', seed)
	d2.write_bandit_data('./', 'Brute', seed)

	d3 = Dataset()
	a = d3.read_bandit_data('./', 'MultiClass')
	b = d3.read_bandit_data('./', 'MultiLabel')
	c = d3.read_bandit_data('./', 'Brute')
