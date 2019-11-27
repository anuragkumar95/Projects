from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

class LogisticRegression:
	def __init__(self):
		self.w = []

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def h(self, X, w):
		return np.dot(X, w.T) 

	def cost(self, h, y):
		cost = y*np.log(self.sigmoid(h)+0.0000000000001) + (1-y)*np.log(1 - self.sigmoid(h)+0.0000000000001)
		return -sum(cost)/len(y)

	def gradient(self, h, X, y):
		grads = (self.sigmoid(h) - y)*X
		return np.sum(grads, axis=0)

	def fit(self, X, y, steps, lr=0.001):
		self.w = np.random.randn(1, X.shape[1])
		J = []
		for i in range(steps):
			h = self.h(X, self.w)
			J.append(self.cost(h, y))
			if i%1000 == 0:
				print(J[-1])
			self.w = self.w - lr*self.gradient(h, X, y)
		return J

	def predict(self, X):
		h = self.h(X, self.w)
		y = []
		for i in h:
			if i >= 0.5:
				y.append(1)
			else:
				y.append(0)
		return y


	def score(self, y, Y):
		count = 0
		for i in range(len(Y)):
			if Y[i] == y[i]:
				count += 1
		return count/len(Y)


X, y = datasets.make_moons(n_samples = 1000, noise = 0.15)
Xtest, ytest = datasets.make_moons(n_samples = 500, noise = 0.15)
const = np.ones((1000,1))
X = np.column_stack((X,\
		     X**2,\
		     X**3,\
		     X[:,0]+X[:,1],\
		     X[:,0]-X[:,1],\
		     X[:,0]*X[:,1],\
		     X[:,0]**2+X[:,1]**2,\
		     X[:,0]**2-X[:,1]**2,\
		     X[:,0]**3+X[:,1]**3,\
		     X[:,0]**3-X[:,1]**3))

Xtest = np.column_stack((Xtest,\
			 Xtest**2,\
			 Xtest**3,\
			 Xtest[:,0]+Xtest[:,1],\
			 Xtest[:,0]-Xtest[:,1],\
			 Xtest[:,0]*Xtest[:,1],\
			 Xtest[:,0]**2+Xtest[:,1]**2,\
			 Xtest[:,0]**2-Xtest[:,1]**2,\
			 Xtest[:,0]**3+Xtest[:,1]**3,\
			 Xtest[:,0]**3-Xtest[:,1]**3))

y = y.reshape(len(y),1)
col = ['red', 'blue']
clf = LogisticRegression()
J = clf.fit(X, y, 10000, 0.001)
y_pred = clf.predict(Xtest)
print("Score:", clf.score(y_pred, ytest))
plt.figure()
plt.plot(J)
plt.figure()
plt.scatter(Xtest[:,0],Xtest[:,1], c=y_pred, cmap = matplotlib.colors.ListedColormap(col))
plt.figure
plt.show()








