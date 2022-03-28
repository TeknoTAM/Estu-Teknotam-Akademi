import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_result = 1 / (1 + np.exp(-z))
    return y_result

def propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_result = sigmoid(z)
    
    #backward propagation
    loss = - y_train * np.log(y_result) - (1 -y_train)*np.log(1- y_result)
    cost = (np.sum(loss))/ x_train.shape[1]
    derivative_weight = (np.dot(x_train, ((y_result - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_result - y_train)/ x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost,gradients

def update(w,b,x_train,y_train,learning_rate,num_iter):

    cost_list = []
    index = []

    for i in range(num_iter):
        cost,gradients = propagation(w,b,x_train,y_train)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate / gradients["derivative_bias"]

        if i % 10 == 0:
            index.append(i)
            cost_list.append(cost)
            print("Cost after iteration {}: {}".format(i,cost))

    parameters = {"weight":w,"bias":b}
    plt.plot(index,cost_list)
    plt.xticks(index,rotation="vertical")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters,cost_list

def predict(w,b,x_test):
    
    z = np.dot(w.T,x_test) + b #w =(4096,1), x_test=(4096,62)  
    y_pred = sigmoid(z)
    for i in range(y_pred.shape[1]):
        if y_pred[0,i] <= 0.5:
            y_pred[0,i] = 0
        else:
            y_pred[0,i] = 1

    return y_pred

def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iter):
    w,b = initialize_weights_and_bias(x_train.shape[0])
    parameters, cost_list = update(w,b,x_train,y_train,learning_rate,num_iter)
    
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    print("Train accuracy: ",100 - np.mean(np.abs(y_prediction_train - y_train)) * 100)
    print("Test accuracy: ",100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)

    return parameters

if __name__ == "__main__":

    X = np.load("sources/digit_dataset/digit_x.npy")
    Y = np.load("sources/digit_dataset/digit_y.npy")
    print("X shape: ",X.shape)
    print("Y shape: ",Y.shape)


    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=42)
    number_of_train = X_train.shape[0]
    number_of_test = X_test.shape[0]
    
    print("Number of train: ",number_of_train)
    print("Number of test: ",number_of_test)

    X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1] * X_train.shape[2])
    X_test_flatten = X_test.reshape(number_of_test,X_test.shape[1] * X_test.shape[2])
    print("X train flatten",X_train_flatten.shape)
    print("X test flatten",X_test_flatten.shape)


    x_train = X_train_flatten.T
    x_test = X_test_flatten.T
    y_train = Y_train.T
    y_test = Y_test.T

    print("x_train shape: ",x_train.shape)
    print("x_test shape: ",x_test.shape)
    print("y_train shape: ",y_train.shape)
    print("y_test shape: ",y_test.shape)


    parameters = logistic_regression(x_train,y_train,x_test,y_test,0.01,150)


    image = X[0,:,:]
    test_image = image.reshape((image.shape[0]*image.shape[1],1))
    y_pred = predict(parameters["weight"],parameters["bias"],test_image)
    image = image * 255

    print("Prediction: ",y_pred)
    plt.figure()
    plt.imshow(image)
    plt.show()

    

    #logistic regresssion sklearn
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train.T,y_train.T)