import numpy
import scipy.misc
import scipy.sparse
import scipy.special
import sys


class Instance:
	def __init__(self, instance_type):
		self.instanceType = instance_type
		self.unset = True

	def set(self, propensity, loss, x, y):
		if propensity <= 0:
			print("Instance:set\t[ERR]\tInvalid propensity ", propensity, flush=True)
			sys.exit(0)

		self.invLogPropensity = -numpy.log(propensity)
		self.loss = loss
		self.x = x
		self.y = y

		self.unset = False


class Brute(Instance):
	def __init__(self, num_features):
		Instance.__init__(self, 'Brute')
		self.numFeatures = num_features

	def parametrize(self):
		weights = numpy.zeros((1,self.numFeatures), dtype=numpy.longdouble)
		return weights

	def risk_gradient(self, weights, clip, compute_gradient):
		if self.unset:
			print("Brute:risk_gradient\t[ERR]\tSet loss, propensity, x, y first", flush=True)
			sys.exit(0)

		scores = numpy.dot(weights, self.x)
		scores = scores.ravel()
		partition = scipy.misc.logsumexp(scores)

		logProbability = scores[self.y] - partition
		importanceWeight = numpy.exp(logProbability + self.invLogPropensity)

		clipped = False
		if (clip > 0) and (importanceWeight > clip):
			importanceWeight = clip
			clipped = True

		risk = self.loss * importanceWeight

		gradient = None
		if (not clipped) and compute_gradient:
			gradient = numpy.zeros(weights.shape, dtype = numpy.longdouble)
			probabilityPerY = numpy.exp(scores - partition)
			numY = numpy.shape(self.x)[1]
			for i in range(numY):
				gradient[0,:] += -risk * probabilityPerY[i] * self.x[:,i]
			gradient[0,:] += risk * self.x[:,self.y]
		return risk, gradient



if __name__ == "__main__":
	a = MultiClass(5, 20)
	a_weights = a.parametrize()
	a_x = numpy.ones(20, dtype = numpy.longdouble)
	a.set(0.1, 1.0, a_x, 3)
	print("Multiclass risk/gradient", a.risk_gradient(a_weights, 5.0, True))

	b = MultiLabel(4, 10)
	b_weights = b.parametrize()
	b_x = numpy.ones(10, dtype = numpy.longdouble)
	b.set(0.1, 1.0, b_x, numpy.array([0,0,1,1], dtype = numpy.int))
	print("Multilabel risk/gradient", b.risk_gradient(b_weights, 5.0, True))

	c = Brute(30)
	c_weights = c.parametrize()
	c_x = numpy.zeros((30,2), dtype = numpy.longdouble)
	c_x[0:15,0] = 1
	c_x[15:,1] = 1
	c.set(0.1, 1.0, c_x, 1)
	print("Brute risk/gradient", c.risk_gradient(c_weights, 5.0, True))
