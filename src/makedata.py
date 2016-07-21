import copy
import itertools
import numpy as np

numInstances = 5
numProductFeatures = 2
numSlots = 2
numCandidates = 3
numInstanceFeatures = numProductFeatures * numSlots
numActions = 1
for i in range(numSlots):
	numActions *= numCandidates - i


class Product:
	def __init__(self, id):
		self.id = id
		self.features = np.random.rand(numProductFeatures) * 10

	def get_feature(self, slotId):
		return self.features / max(slotId, 1)


class Action:

	def __init__(self, aid, products):
		self.id = aid
		self.products = products

		self.features = []
		for i in range(numSlots):
			product = products[i]
			slotId = i + 1
			self.features.extend(product.get_feature(slotId))

		self.score = sum(self.features)

	def set(self, propensity):
		pass


class Banner:
	def __init__(self, bid):
		self.id = bid
		self.candidates = [Product(cid) for cid in range(numCandidates)]
		self.actions = [Action(aid, permut) for aid, permut in enumerate(itertools.permutations(self.candidates, numSlots))]


class DecisionMaker:
	def __init__(self):
		self.banners = [Banner(bid) for bid in range(numInstances)]

	def assign(self):
		pass


if __name__ == '__main__':
	banner = Banner()
	for action in banner.actions:
		for p in action.products:
			print p.features
		print action.features
		print action.score
