#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def user_bias_update(A, u, v, mu, c):
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    b = np.array([0.] * m)
    for i in xrange(m):
        for j in xrange(n):
            b[i] += (-1. / n) * (np.dot(u[i], v[j].T) + c[j] + mu - A[i, j])
    return b


def user_vector_update(A, v, k, mu, b, c):
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    u = np.zeros([m, k])
    v_matrix = np.dot(v.T, v)
    for i in xrange(m):
        right_side = np.zeros([k, ])
        for j in xrange(n):
            right_side += b[i] * v[j]
            right_side += c[j] * v[j]
            right_side += mu * v[j]
            right_side -= A[i, j] * v[j]
        u[i, :] = -np.dot(np.linalg.inv(v_matrix), right_side)
    return u


def movie_bias_update(A, u, v, mu, b):
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    c = np.array([0.] * n)
    for i in xrange(m):
        for j in xrange(n):
            c[j] += (-1. / m) * (np.dot(u[i], v[j].T) + b[i] + mu - A[i, j])
    return c


def movie_vector_update(A, u, k, mu, b, c):
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    v = np.zeros([n, k])
    u_matrix = np.dot(u.T, u)
    for j in xrange(n):
        right_side = np.zeros([k, ])
        for i in xrange(m):
            right_side += b[i] * u[i]
            right_side += c[j] * u[i]
            right_side += mu * u[i]
            right_side -= A[i, j] * u[i]
        v[j, :] = -np.dot(np.linalg.inv(u_matrix), right_side)
    return v


def log_update(A, u, v, T, mu, b, c):
    log_iter = 0
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    for i in xrange(m):
        for j in xrange(n):
            log_iter += (-1. / 2) * \
                ((np.dot(u[i], v[j].T) + b[i] + c[j] + mu - A[i, j])**2)
    return log_iter


def alt_least_squares(A, k, T):
    """
    Inputs:
    A: input data
    k: number of dimensions for movie vectors & user vectors
    T: number of iterations

    Output:
    Log-likelihood function for each iteration
    """
    # Calculate average rating in A
    mu = np.mean(A["ratings"])

    # Independently draw u_i and v_j vectors from multivariate normal
    m = max(A["i"])
    n = max(A["j"])
    omega = len(A["i"])
    # Total # of elements in matrix
    A_matrix = np.zeros([m, n])
    for l in xrange(omega):
        A_matrix[A["i"][l] - 1][A["j"][l] - 1] = A["ratings"][l]
    mean_vect = np.array([0] * k)
    cov_matrix = (1. / k) * np.identity(k)
    u = []
    v = []
    for i in xrange(m):
        u.append(np.random.multivariate_normal(mean_vect, cov_matrix))
    for j in xrange(n):
        v.append(np.random.multivariate_normal(mean_vect, cov_matrix))
    # Initalize b_i and c_j to 0
    u = np.array(u)
    v = np.array(v)
    b = np.array([0.] * m)
    c = np.array([0.] * n)
    # for all iterations
    log_like = np.zeros([T])
    for t in xrange(T):
        # update b_i
        b = user_bias_update(A_matrix, u, v, mu, c)
        # update u_i
        u = user_vector_update(A_matrix, v, k, mu, b, c)
        # update c_j
        c = movie_bias_update(A_matrix, u, v, mu, b)
        # update v_j
        v = movie_vector_update(A_matrix, u, k, mu, b, c)
        # update log-likelihood
        log_like[t] = log_update(A_matrix, u, v, T, mu, b, c)
    return log_like

if __name__ == '__main__':
    ratings_fake = pd.read_csv("ratings_fake.csv",
                               quotechar='"', encoding='UTF-8',
                               names=["i", "j", "ratings"])
    n_features = 5
    n_iter = 20

    log_fake = alt_least_squares(ratings_fake, n_features, n_iter)

    plt.plot(range(20), log_fake)
    plt.axis([0, 25, min(log_fake) - 100, max(log_fake) + 100])
    plt.xlabel('Iteration Number')
    plt.ylabel('Log-Likelihood')
    plt.title('Log Likelihood of Alternating Least Squares')
    plt.show()
    plt.close()
