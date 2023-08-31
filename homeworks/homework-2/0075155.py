import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X_train = np.genfromtxt(fname = "hw02_data_points.csv", delimiter = ",", dtype = float)
y_train = np.genfromtxt(fname = "hw02_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below

    k = 5

    class_priors = [(np.mean([y == (c + 1)])) for c in range(k)]
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_class_means(X, y):
    # your implementation starts below
    k = 5

    lst = []

    for i in range(k):
        x1 = np.mean(X[y == i + 1], axis = 0)  
        
        lst.append(x1)

    sample_means = np.stack(lst)

    # your implementation ends above
    return(sample_means)

sample_means = estimate_class_means(X_train, y_train)
print(sample_means)



# STEP 5
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_class_covariances(X, y):
    # your implementation starts below
    k = 5
    lst = []

    for i in range(k):
        part3 = np.array([[0,0],
                 [0,0]])
        for j in range(1000):
            part3 = np.add(np.matmul(np.transpose([X[y == (i+1)][j] - sample_means[i]]),
                        np.transpose(np.transpose([X[y == (i+1)][j] - sample_means[i]]))), part3)
        part3 = part3 * (1/1000)

        test = [[part3[0][0],part3[0][1]],
                [part3[1][0],part3[1][1]]] 
        lst.append(test)

    sample_covariances = np.stack(lst)
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_class_covariances(X_train, y_train)
print(sample_covariances)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, class_means, class_covariances, class_priors):
    # your implementation starts below
    
    D = 2
    k = 5
    nums = X.shape[0]
    val1 = -0.5 * (D * np.log(math.pi * 2))

    lst = []
    for i in range(nums):
        lst2 = []
        for j in range(k):
            val2 = -0.5 * np.log(linalg.det(class_covariances[j]))
            val3_2 = linalg.inv(class_covariances[j])
            val4 = np.log(class_priors[j])

            val3_1 = (X[i] - class_means[j])
            intval = np.matmul(val3_1.T,val3_2)
            intval = np.matmul(intval,val3_1)
            val3 = -0.5 * intval
            score_values = val1 + val2 + val3 + val4
            lst2.append(score_values)
        lst.append(lst2)

    score_values = np.stack(lst)

    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below

    confusion_matrix = np.zeros((5,5), dtype = int)
    sizeY = y_truth.shape[0]
    for i in range(sizeY):
        confusion_matrix[np.argmax(scores[i])][y_truth[i] - 1] +=1
        
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)



def draw_classification_result(X, y, class_means, class_covariances, class_priors):
    class_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    K = np.max(y)

    x1_interval = np.linspace(-75, +75, 151)
    x2_interval = np.linspace(-75, +75, 151)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    scores_grid = calculate_score_values(X_grid, class_means, class_covariances, class_priors)

    score_values = np.zeros((len(x1_interval), len(x2_interval), K))
    for c in range(K):
        score_values[:,:,c] = scores_grid[:, c].reshape((len(x1_interval), len(x2_interval)))

    L = np.argmax(score_values, axis = 2)

    fig = plt.figure(figsize = (6, 6))
    for c in range(K):
        plt.plot(x1_grid[L == c], x2_grid[L == c], "s", markersize = 2, markerfacecolor = class_colors[c], alpha = 0.25, markeredgecolor = class_colors[c])
    for c in range(K):
        plt.plot(X[y == (c + 1), 0], X[y == (c + 1), 1], ".", markersize = 4, markerfacecolor = class_colors[c], markeredgecolor = class_colors[c])
    plt.xlim((-75, 75))
    plt.ylim((-75, 75))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    return(fig)
    
fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_different_covariances.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_shared_class_covariance(X, y):
    # your implementation starts below

    mean = np.mean(X, axis = 0)

    arr = [[0,0],
           [0,0]]
    for i in range(X.shape[0]):
        arr = np.add((np.matmul( np.transpose([(X[i] - mean)]),
                                            np.transpose(np.transpose([(X[i] - mean)])) ) ), arr)
    lst = [[arr[0][0]/5000,arr[0][1]/5000],
                          [arr[1][0]/5000,arr[1][1]/5000]]
    lst2 = []
    for i in range(5):
        lst2.append(lst)
    sample_covariances = np.stack(lst2)
    
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_shared_class_covariance(X_train, y_train)
print(sample_covariances)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_shared_covariance.pdf", bbox_inches = "tight")
