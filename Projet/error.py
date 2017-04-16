import numpy as np
with open("res.txt") as f:
    content = f.readlines()


a = np.zeros(101)
b = np.zeros(101)
# b += 1
for i in content:
	i = i.strip()
	data = i.split(";")
	print(data)
	if int(data[1]) > b[int(data[0])]:
		a[int(data[0])] += int(data[1])
		b[int(data[0])] += int(data[1])

f = open("res3.txt","w")
for i in range(len(a)):
	# a[i] /= b[i]
	f.write(str(a[i])+"\n")


#res.txt : age;err abs; err signed
#res2.txt : pr chaque age : error moy
#res3.txt : pr chaque age : error max