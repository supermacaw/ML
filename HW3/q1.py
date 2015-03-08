import numpy, scipy
import matplotlib.pyplot as plt

#print numpy.random.normal(3,9)

total = 0
numbers = []
for i in range(100):
	sub = numpy.random.normal(3,3)
	total += sub
	numbers.append(sub)
print total / float(100)

numbers2 = []
total2 = 0
for i in range(100):
	sub2 = 0.5 * numbers[i] + numpy.random.normal(4,2)
	total2+=sub2
	numbers2.append(sub2)

print total2/float(100)
a = numpy.cov(numbers,numbers2)
print a 
#a[1,0] = 0
b = numpy.linalg.eig(a)
x1s= [3]
x2s = [4]
print b
for i in range(len(b[1])):
	x1s.append((b[1][1-i][0]) *b[0][i] + 3)
	x2s.append((b[1][1-i][1])* b[0][i] + 4)
	x1s.append(3)
	x2s.append(4)

#plt.scatter(numbers, numbers2)
#plt.plot(x1s,x2s)
#plt.axis('equal')
#plt.show()

points = numpy.asarray(zip(numpy.asarray(numbers)-3, numpy.asarray(numbers2)-4))
#print points
new_points1 = []
new_points2 = []
for point in points:
	new_points1.append(numpy.dot(point, b[1][0].transpose()))
	new_points2.append(numpy.dot(point, b[1][1].transpose()))
#print zip(new_points1, new_points2)

plt.scatter(new_points1,new_points2)
plt.axis([-15,15,-15,15])
#plt.show()

