import numpy as np
import pandas as pd



X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below

    n = 50000

    X_train = X[:n]
    y_train = y[:n]
    X_test = X[n:]
    y_test = y[n:]
    
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below

    k = 2

    class_priors = [(np.mean([y == (c + 1)])) for c in range(k)]
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    
    k = 2

    yk = [(y == (c + 1)).astype(int) for c in range(k)]
    nk = [yc.sum() for yc in yk]
    transform = [yk[i] / nk[i] for i in range(k)]

    pAcd = np.matmul(transform, X == "A")
    pCcd = np.matmul(transform, X == "C")
    pGcd = np.matmul(transform, X == "G")
    pTcd = np.matmul(transform, X == "T")
    
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    lst = []

    for i in range(len(X)):
        s1 = 0
        s2 = 0

        pa1 = 0 
        pc1 = 0
        pg1 = 0
        pt1 = 0
        pa2 = 0 
        pc2 = 0
        pg2 = 0
        pt2 = 0
        for j in range(len(pAcd[0])):
            if(X[i][j] == "A"):
                pa1 += np.log(pAcd[0][j])
                pa2 += np.log(pAcd[1][j])
            if(X[i][j] == "C"):
                pc1 += np.log(pCcd[0][j])
                pc2 += np.log(pCcd[1][j])
            if(X[i][j] == "G"):
                pg1 += np.log(pGcd[0][j])
                pg2 += np.log(pGcd[1][j])
            if(X[i][j] == "T"):
                pt1 += np.log(pTcd[0][j])
                pt2 += np.log(pTcd[1][j])
        s1 = pa1+pc1+pg1+pt1 + np.log(class_priors[0])
        s2 = pa2+pc2+pg2+pt2 + np.log(class_priors[1])
        lst.append([s1,s2])

    score_values = np.stack(lst)
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    confusion_matrix = np.matrix([[0,0]
                                 ,[0,0]])

    for i in range(len(y_truth)):
        actual = y_truth[i]
        model1 = scores[i][0]
        model2 = scores[i][1]

        if   (model1 > model2) and (actual == 1):
            confusion_matrix[0,0] += 1
        elif (model1 < model2) and (actual == 2):
            confusion_matrix[1,1] += 1
        elif (model1 > model2) and (actual == 2):
            confusion_matrix[0,1] += 1
        else:
            confusion_matrix[1,0] += 1



    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
