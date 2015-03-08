import scipy.io, numpy as np, sklearn.svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt 
import scipy.stats
import math

def compute(train_size):
	#loading in the data
	train_set=scipy.io.loadmat('./data/spam-dataset/spam_data.mat')
	subset_train = np.random.choice(5172, train_size, replace=False)
	subset_validation = np.random.choice(5172, 1000, replace=False)
	train_data=train_set['training_data'][subset_train,:]
	train_labels = train_set['training_labels'][:,subset_train]

	test_data = train_set['test_data']
	test_labels = train_set['training_labels'][:,subset_validation]

	kaggle_data = train_set['test_data']

	index_of_classes = {0:[],1:[]}
	class_data = {0:[],1:[]}
	index = 0
	for i in train_labels[0,:]:
		index_of_classes[i].append(index)
		#size = np.linalg.norm(np.reshape(train_images[:,:,index],(784,)))
		class_data[int(i)].append(train_data[index,:])#/float(size))
		index+=1

	#data has been processed
	#perform MLE
	means = {i:np.zeros((32,)) for i in range(2)}
	variances = {i:np.zeros((32,32)) for i in range(2)}

	"""for i in range(10):
		for j in class_data[i]:
			means[i] += j
		means[i] = means[i]/float(len(class_data[i]))

	for i in range(10):
		for j in class_data[i]:
			#print np.outer((j - means[i]),np.transpose(j-means[i]))
			variances[i] =  np.outer((j - means[i]),np.transpose(j-means[i]))
		#print float(len(class_data[i]))
		variances[i] = variances[i]/float(len(index_of_classes[i]))"""
	for i in range(2):
		class_data[i] = np.rollaxis(np.asarray(class_data[i]),1,0)
		means[i] = np.mean(class_data[i],1)
		#print means[i].shape
		variances[i] = np.cov(class_data[i]) + 0.0001*scipy.sparse.identity(32)
		#plt.imshow(variances[i])
		#plt.show()

	#calculate class frequencies
	frequencies = {i:0 for i in range(2)}
	for i in range(2):
		frequencies[i] = float(len(class_data[i]))/float(train_size)

	mean_var = np.zeros((32,32))
	for i in range(2):
		mean_var += variances[i]
	mean_var = mean_var / float(2)
	mean_var = mean_var + 0.0001*scipy.sparse.identity(32)

	predictions = []

	#print scipy.stats.multivariate_normal(means[0], mean_var).logpdf([np.reshape(test_images[:,:,0],(784,)), np.reshape(test_images[:,:,1],(784,))])

	#for n in range(10):

	model0 = scipy.stats.multivariate_normal(means[0], mean_var)#variances[0])
	model1 = scipy.stats.multivariate_normal(means[1], mean_var)#variances[1])

	for i in range(5857):
		to_use = test_data[i,:]
		probs = {i:None for i in range(2)}
		probs[0] = model0.logpdf(to_use) + math.log(frequencies[0])
		probs[1] = model1.logpdf(to_use) + math.log(frequencies[1])
		predictions.append(max(probs, key=probs.get))
	#count=0
	#for i in range(1000):
#		if test_labels[0,i] == predictions[i]:
#			count+=1
#	print train_size
	#return float(count)/float(1000)
	print predictions
	f = open('spam_predictions.csv','w')
	f.write('Id,Category\n')
	for i in range(5857):
		f.write(str(i+1)+","+str(predictions[i])+'\n')

compute(5172)
#success_rates = []
#for size in [5172]:
#	success_rates.append(compute(size))
#print success_rates
#plt.plot([100,200,500,1000,2000], success_rates)
#plt.show()