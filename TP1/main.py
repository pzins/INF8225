
class Node:
	'Class representing a node in a graph of factors (bayes net)'

	def __init__(self, name, proba):
		self.name = name
		self.neighbours = []
		self.proba = proba
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

	def __init__(self, name, proba):
		super().__init__(name, proba)

	def getMessage(self):
		self.visited = True
		if len(self.neighbours) == 1 and self.name != "x3":
			return self.proba
		st = ""
		for i in self.neighbours:
			if not i.visited:
				st += i.getMessage() + " "
		return st

class Function(Node):
	'Class representing a function in a graph of factor (bayes net)'

	def __init__(self, name, proba):
		super().__init__(name, proba)
	
	def getMessage(self):
		self.visited = True
		if len(self.neighbours) == 1 and self.name != "x3":
			return self.proba
		st = ""
		for i in self.neighbours:
			if not i.visited:
				st += self.proba + " " + i.getMessage() + " "
		return st

class Message:
	'Class representing a message'

	def __init__(self, origin, dest, typeM):
		self.origin = origin
		self.dest = dest
		self.typeM = typeM

	def __str__(self):
		return origin.name+"->"+dest.name



fa = Function("fa", "SOMME_ P(x1)")
fb = Function("fb", "SOMME_ P(x2)")
fc = Function("fc", "SOMME_ P(x3|x1)")
fd = Function("fd", "SOMME_ P(x4|x1, x2)")
fe = Function("fe", "SOMME_ P(x5|x2)")

x1 = Variable("x1", "")
x2 = Variable("x2", "")
x3 = Variable("x3", "")
x4 = Variable("x4", "")
x5 = Variable("x5", "1")

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

print(x3.getMessage())