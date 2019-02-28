import numpy as np
import math
import matplotlib.pyplot as plt

#This function returns the gaussian probabilty value for point x.
def Gaussian(x,u,sigma):
	pow_ = (x-u)/sigma
	return np.exp(-0.5*(pow_**2))/np.sqrt(2*3.14*(sigma**2))

#returns the effective number of points 'N' assigned to each class	 
def expectation(x,num_clusters,p_clusters,u_clusters,sigma_clusters):
	expectation = []
	for c in range(num_clusters):
		exp = [Gaussian(x[i],u_clusters[c],sigma_clusters[c])*p_clusters[c] for i in range(len(x))]
		expectation.append(exp)
	expectation = np.asarray(expectation)
	sum_ = np.sum(expectation,axis = 0)
	expectation = [i/sum_ for i in expectation]
	N = [sum(x) for x in expectation]
	return N,expectation

#main algorithm function
def Expectation_Maximization(x,num_clusters,p_clusters,u_clusters,sigma_clusters):
	L = [-9999999]
	change = 1000

	while(abs(change) > 0.001):
		l = 0
		l_hood = 0
		N, exp = expectation(x,num_clusters,p_clusters,u_clusters,sigma_clusters)
		for c in range(num_clusters):
			#update mean
			u_clusters[c] = sum([exp[c][i]*x[i] for i in range(len(x))])/N[c]
			e_u2 = sum([exp[c][i]*(x[i]**2) for i in range(len(x))])/N[c]
			new_var = (e_u2) - (u_clusters[c]**2) 
			#update sigma
			sigma_clusters[c] = np.sqrt(new_var)
			#update P(c)
			p_clusters[c] = N[c]/sum(N)
			
    	#calculating loglikelihood
		for i in range(len(x)):
			for c in range(num_clusters):
				l += p_clusters[c]*Gaussian(x[i],u_clusters[c],sigma_clusters[c])	
			l_hood += np.log(l)
		change = l_hood - L[-1] 
		L.append(l_hood)
	return L[1:]

if __name__ == '__main__':
	#Reading data
	inp = open("data1.txt", encoding = 'utf-8')
	x = [float(i) for i in inp.read().split('\n') if len(i)>=1]
	min_,max_ = min(x),max(x)
	#Initializing parameteres..
	num_clusters = 3
	p_clusters = np.ones(num_clusters)/num_clusters
	u_clusters = np.linspace(min_, max_, num_clusters).reshape(num_clusters, 1)
	sigma_clusters = min_ + np.random.random([num_clusters, 1]) * (max_ - min_) / 10
	#Calculating loglikelihood
	loglikelihood = Expectation_Maximization(x,num_clusters,p_clusters,u_clusters,sigma_clusters)
	print("Mu:",u_clusters.reshape(num_clusters))
	print("Sigma:",sigma_clusters.reshape(num_clusters))
	print("probabilty:",p_clusters)
	#Plotting the histogram, we can see the gaussians mixtures are centered around 0.6, 0.3 and 0.8
	#which is similar to the approximated means we are getting after EM with 3 clusters.
	f1 = plt.figure()
	plt.hist(x,bins=30)
	f2 = plt.figure()
	plt.plot(loglikelihood)
	plt.show()
	
