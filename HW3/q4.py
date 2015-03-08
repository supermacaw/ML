import scipy.io, numpy as np, sklearn.svm
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt 
import scipy.stats
import math

def compute(train_size):
	#loading in the data
	train_data=scipy.io.loadmat('./data/digit-dataset/train.mat')
	subset_train = np.random.choice(60000, train_size, replace=False)
	train_images=train_data['train_image'][:,:,subset_train]
	train_labels = train_data['train_label'][subset_train]

	test_data = scipy.io.loadmat('./data/digit-dataset/test.mat')
	test_images = test_data['test_image']
	test_labels = test_data['test_label']

	kaggle_data = scipy.io.loadmat('./data/digit-dataset/kaggle.mat')
	kaggle_images = kaggle_data['kaggle_image']

	index_of_classes = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
	class_data = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
	index = 0
	for i in train_labels:
		index_of_classes[int(i)].append(index)
		size = np.linalg.norm(np.reshape(train_images[:,:,index],(784,)))
		class_data[int(i)].append(np.reshape(train_images[:,:,index],(784,))/float(size))
		index+=1

	#data has been processed
	#perform MLE
	means = {i:np.zeros((784,)) for i in range(10)}
	variances = {i:np.zeros((784,784)) for i in range(10)}

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
	for i in range(10):
		class_data[i] = np.rollaxis(np.asarray(class_data[i]),1,0)
		means[i] = np.mean(class_data[i],1)
		#print means[i].shape
		variances[i] = np.cov(class_data[i]) + 0.0001*scipy.sparse.identity(784)
		plt.imshow(variances[i])
		plt.show()

	#calculate class frequencies
	frequencies = {i:0 for i in range(10)}
	for i in range(10):
		frequencies[i] = float(len(class_data[i]))/float(train_size)

	mean_var = np.zeros((784,784))
	for i in range(10):
		mean_var += variances[i]
	mean_var = mean_var / float(10)
	mean_var = mean_var + 0.0001*scipy.sparse.identity(784)

	predictions = []

	#print scipy.stats.multivariate_normal(means[0], mean_var).logpdf([np.reshape(test_images[:,:,0],(784,)), np.reshape(test_images[:,:,1],(784,))])

	#for n in range(10):

	model0 = scipy.stats.multivariate_normal(means[0], mean_var)#variances[0])
	model1 = scipy.stats.multivariate_normal(means[1], mean_var)#variances[1])
	model2 = scipy.stats.multivariate_normal(means[2], mean_var)#variances[2])
	model3 = scipy.stats.multivariate_normal(means[3], mean_var)#variances[3])
	model4 = scipy.stats.multivariate_normal(means[4], mean_var)#variances[4])
	model5 = scipy.stats.multivariate_normal(means[5], mean_var)#variances[5])
	model6 = scipy.stats.multivariate_normal(means[6], mean_var)#variances[6])
	model7 = scipy.stats.multivariate_normal(means[7], mean_var)#variances[7])
	model8 = scipy.stats.multivariate_normal(means[8], mean_var)#variances[8])
	model9 = scipy.stats.multivariate_normal(means[9], mean_var)#variances[9])

	for i in range(5000):
		to_use = np.reshape(test_images[:,:,i],(784,))
		probs = {i:None for i in range(10)}
		probs[0] = model0.logpdf(to_use) + math.log(frequencies[0])
		probs[1] = model1.logpdf(to_use) + math.log(frequencies[1])
		probs[2] = model2.logpdf(to_use) + math.log(frequencies[2])
		probs[3] = model3.logpdf(to_use) + math.log(frequencies[3])
		probs[4] = model4.logpdf(to_use) + math.log(frequencies[4])
		probs[5] = model5.logpdf(to_use) + math.log(frequencies[5])
		probs[6] = model6.logpdf(to_use) + math.log(frequencies[6])
		probs[7] = model7.logpdf(to_use) + math.log(frequencies[7])
		probs[8] = model8.logpdf(to_use) + math.log(frequencies[8])
		probs[9] = model9.logpdf(to_use) + math.log(frequencies[9])
		predictions.append(max(probs, key=probs.get))
	count=0
	for i in range(5000):
		if test_labels[i] == predictions[i]:
			count+=1
	print train_size
	return float(count)/float(5000)
#f = open('digit_predictions.csv','w')
#f.write('Id,Category\n')
#for i in range(5000):
#	f.write(str(i+1)+","+str(predictions[i])+'\n')
success_rates = []
for size in [100,200,500,1000,2000,5000,10000,30000,60000]:
	success_rates.append(1-compute(size))
print success_rates
plt.plot([100,200,500,1000,2000,5000,10000,30000,60000], success_rates)
plt.show()