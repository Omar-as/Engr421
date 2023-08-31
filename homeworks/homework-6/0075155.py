import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[+0.0, +5.5],
                        [+0.0, +0.0],
                        [+0.0, -5.5]])

group_covariances = np.array([[[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+3.2, +2.8],
                               [+2.8, +3.2]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw06_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 3

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    
    means = np.genfromtxt("hw06_initial_centroids.csv", delimiter=',')
    covariances = np.array([np.cov(X, rowvar=False)] * K)
    priors = np.full((K, ), fill_value=1/K)

    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below

    assignments = np.zeros(X.shape[0])
    for _ in range(100):

        responsibilities = np.zeros((X.shape[0], K))
        for i in range(K):
            responsibilities[:, i] = priors[i] * stats.multivariate_normal(means[i], covariances[i]).pdf(X)
        sum_responsibilities = np.sum(responsibilities, axis=1).reshape(-1,1)
        responsibilities /= sum_responsibilities
        
        for i in range(K):
            responsibility = responsibilities[:, i].reshape(-1,1)
            total_responsibility = np.sum(responsibility)
            means[i] = np.sum(responsibility * X, axis=0) / total_responsibility
            deviation = X - means[i]
            covariances[i] = (deviation.T @ (responsibility * deviation)) / total_responsibility
            priors[i] = total_responsibility / X.shape[0]
        assignments = np.argmax(responsibilities, axis=1)

    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below

    plt.figure(figsize=(7,7))
    color = ['b', 'g', 'r']
    
    for i in range(K):
        plt.scatter(X[assignments == i, 0], X[assignments == i, 1], s=10, color=color[i])

    x = np.linspace(-8, 8, 151)
    y = np.linspace(-8, 8, 151)

    X_mesh, Y_mesh = np.meshgrid(x, y)

    pos = np.empty(X_mesh.shape + (2,))
    pos[:, :, 0] = X_mesh; pos[:, :, 1] = Y_mesh

    for i in range(K):
        plt.contour(X_mesh, Y_mesh, stats.multivariate_normal(means[i], covariances[i]).pdf(pos), levels=[0.01], colors=color[i])
        plt.contour(X_mesh, Y_mesh, stats.multivariate_normal(group_means[i], group_covariances[i]).pdf(pos), levels=[0.01], colors='k', linestyles='dashed')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.legend()
    plt.show()
    plt.savefig("Clustering_results.pdf")
    
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)

