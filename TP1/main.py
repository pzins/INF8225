import re

class Node:
	'Class representing a node in a graph of factors (bayes net)'

	def __init__(self, name):
		self.name = name
		self.neighbours = []
		self.visited = False

	def addNeighbours(self, neighbours):
		for i in neighbours:
			self.neighbours.append(i)
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
	'Class representing a variable in a graph of factors (bayes net)'

	def __init__(self, name):
		super().__init__(name)

	def getMessage(self, caller, observed):
		self.visited = True
		if len(self.neighbours) == 1 and self.name != "x3":
			return "1"
		st = ""
		for i in self.neighbours:
			if not i.visited:
				st += i.getMessage(self.name, observed) + " "
		return st

class Function(Node):
	'Class representing a function in a graph of factor (bayes net)'

	def __init__(self, name, proba):
		super().__init__(name)
		self.proba = proba
		tmp=proba
		for i in ["P","(",")",","," ","|"]:
			tmp = tmp.replace(i, "")
		self.proba_args = re.findall('..', tmp)



			

	def getMessage(self, caller, observed):
		self.visited = True
		# if function node is a leaf => no sum
		if len(self.neighbours) == 1 and self.name != "x3":
			return self.proba
		st = ""
		for i in self.neighbours:
			if not i.visited and i not in observed:
				if len(self.proba_args)>1:
					somme_str = "SOMME_"
					for j in self.proba_args:
						if j != caller:
							somme_str += "_"+j
				else:
					somme_str = ""

				st += somme_str + " " + self.proba + " " + i.getMessage(self.name, observed) + " "
		return st


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

mainNode = x3

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

print(x3.getMessage(x3.name, [x4]))