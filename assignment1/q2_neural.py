#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    N = X.shape[0]
    ofs = 0
    Dx, H, Dy = (int(dimensions[0]), int(dimensions[1]),int(dimensions[2]))
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    hidden = sigmoid(X.dot(W1)+b1)
    scores = np.exp(hidden.dot(W2) + b2)
    sum_scores = np.sum(scores,axis = 1)
    correct_scores = scores[labels > 0]
    cost = -np.sum(np.log(correct_scores/sum_scores))/N #+ reg*(np.sum(W1**2) + np.sum(W2**2))
    
    
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    dS = scores/(sum_scores.reshape((-1,1)))
    dS[labels > 0] -= 1
    gradb2 = np.ones(N).dot(dS)/N
    gradW2 = hidden.T.dot(dS)/N# + 2*reg*W2
    dhidden = dS.dot(W2.T)
    gradb1 = np.ones(N).dot(sigmoid_grad(hidden)*dhidden)/N
    gradW1 = X.T.dot(sigmoid_grad(hidden)*dhidden)/N# + 2*reg*W1
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print ("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params :forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
#     your_sanity_checks()
