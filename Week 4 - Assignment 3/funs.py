import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize




def DisplayData(x):
    m = np.empty(200).reshape(-1,1) #
    X = np.random.permutation(x)
    for i in range(10):
        j = np.random.randint(10,499)
        n = X[j*10:10*(j+1),:].reshape(200,20)
        m = np.hstack((m,n))
    fig,ax = plt.subplots(figsize = (20,10)) #arbitrarily chosen figure size
    ax.imshow(m.T, cmap = 'gray')
    plt.show()



def sigmoid(z):
    #define sigmoid
    return 1/(1+np.exp(-z))



def CostFunctionReg(theta,x,y,lamda):
    # For getting the regularized cost 
    
    theta = theta.reshape(-1,1)
    #if x.ndim == 1:
        #return 1
    h = sigmoid(x@theta).reshape(-1,1)
    x = np.hstack((np.ones((x.shape[0],1)), x))
    cost  =  (-1/y.shape[0]) *sum(y*np.log(h) + (1-y)*np.log(1-h))
    cost = cost + (lamda/(2*x.shape[0]))*sum(theta[1:,0]*theta[1:,0])
    return cost



def GradientReg(theta,x,y,lamda):
    # For getting the regularized gradient
    
    theta = theta.reshape(-1,1)
    h = sigmoid(x@theta).reshape(-1,1)
    grad = np.zeros(theta.shape)
    grad = (1/y.shape[0])*np.sum(np.multiply((h-y),x), axis = 0).reshape(-1,1)
    grad[1:,0] = lamda*theta[1:,0]/x.shape[0] + grad[1:,0]
    grad = np.ndarray.flatten(grad)
    return grad



def OneVsAll(x,y, num_labels, lamda):
    # For training the theta for the one vs all classification 
    
    X = np.hstack((np.ones((x.shape[0],1)), x))
    theta_init = np.zeros((X.shape[1],1))
    theta = np.zeros((num_labels,401))
    for i in range(num_labels):
        y_cat = (y == i).astype(int) # with boolean doesn't work, so convert True -> 1
        res = minimize(CostFunctionReg, theta_init, args = (X, y_cat, lamda), jac = GradientReg, method = 'BFGS')
        theta[i,:] = res.x
    return theta



def PredictOneVsAll(theta,x):
    # For predicting the digits from the given data of pixels 
    
    x = np.hstack((np.ones((x.shape[0],1)), x))
    h = sigmoid(x@theta.T)
    pred = np.zeros((h.shape[0],1))
    for i in range(x.shape[0]):
        maxval = np.max(h[i,:])
        maxindex = np.where(h[i,:] == maxval)[0]
        pred[i] = maxindex 
    return pred


def Predict(Theta1, Theta2, x):
    # for returning predicted image label

    x = np.hstack((np.ones(x.shape[0]).reshape(-1,1), x))
    layer2 = sigmoid(x@Theta1.T)
    layer2 = np.hstack((np.ones(layer2.shape[0]).reshape(-1,1), layer2))
    layer3 = sigmoid(layer2@Theta2.T)
    
    pred = np.argmax(layer3, axis = 1).reshape(-1,1) + 1
    return pred