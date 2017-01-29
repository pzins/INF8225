import re

class Node:
	'Class representing a node in a factor graph (bayes net)'

	def __init__(self, name):
		self.name = name
		self.neighbours = []
		self.visited = False

	def addNeighbours(self, neighbours):
		for i in neighbours:
			self.neighbours.append(i)
		# sort neighbours according to their number of neighbours
		self.neighbours.sort(key = lambda x: len(x.neighbours))


	def __str__(self):
		print(self.name)
		n = ""
		print("neighbours = ", end="")
		for i in self.neighbours:
			n += i.name + " "
		print(n)
		print(self.proba)
		return ""		

class Variable(Node):
	'Class representing a variable in a factor graph (bayes net)'

	def __init__(self, name):
		super().__init__(name)
		self.root = False

	def getMessage(self, caller, observed):
		self.visited = True
		if len(self.neighbours) == 1 and not self.root:
			return "1"
		st = ""
		for i in self.neighbours:
			if not i.visited:
				st += i.getMessage(self.name, observed)
		return st

class Function(Node):
	'Class representing a function in a factor graph (bayes net)'

	def __init__(self, name, proba):
		super().__init__(name)
		self.proba = proba
		
		# extract all variables from the function (x1, x2, ...)
		tmp=proba
		tmp = tmp.replace('P',' ')
		self.proba_args = re.findall(r'\w+', tmp)



	def getMessage(self, caller, observed):
		self.visited = True
		# if function node is a leaf => no sum
		if len(self.neighbours) == 1:
			return self.proba
		st = ""
		for i in self.neighbours:
			#one node should not be observed and not considered twice
			if not i.visited and i not in observed:
				# marginalisation : if other variables (x1, x2, ...) are in the function, we compute a sum over these variables
				if len(self.proba_args)>1:
					somme_str = "SOMME_"
					for j in self.proba_args:
						# no sum over the current variable and observed variables
						if j != caller and not j in [k.name for k in observed]:
							somme_str += "_"+j
				st += somme_str + " " + self.proba + " " + i.getMessage(self.name, observed)
		return st

def getProbability(node, observed):
	# we specify this node as the root 
	node.root = True
	return node.getMessage(node.name, observed)

# declaration of the factor graph
fa = Function("fa", "P(x1)")
fb = Function("fb", "P(x2)")
fc = Function("fc", "P(x3|x1)")
fd = Function("fd", "P(x4|x1, x2)")
fe = Function("fe", "P(x5|x2)")

x1 = Variable("x1")
x2 = Variable("x2")
x3 = Variable("x3")
x4 = Variable("x4")
x5 = Variable("x5")

# declaration of the edges
x5.addNeighbours([fe])
fe.addNeighbours([x2, x5])
fb.addNeighbours([x2])
x2.addNeighbours([fd, fe, fb])
fd.addNeighbours([x1, x2, x4])
x4.addNeighbours([fd])
x1.addNeighbours([fc, fa, fd])
fa.addNeighbours([x1])
fc.addNeighbours([x1, x3])
x3.addNeighbours([fc])

# compute probability
print(getProbability(x3, [x4]))