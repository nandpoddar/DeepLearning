## DeepLearning

Today, we will build a logistic regression classifier to recognize cats.  We will use python numpy library for all scientific computation.

First, import all packages - used in this exercise

* numpy
* matplotlib
* h5py
* PIL and Scipy

## Problem statement

You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat. Available datasets (data.h5) contains 
* Train dataset - A training dataset of m_train labled  examples
* Test dataset - A test dataset of m_test labled examples
* Each image is of shape (num_px, num_px, 3) where 3 is for  RBG channels

## Loading and cleaning of data

* Load data - train and test - using code in lr_utils.py
* shape of training data (X) - (209,64,64,3)
* shape of test data (X) - (50, 64,64,3)
* shape of training data - (Y) - (1,209)
* shape of training data - (Y) - (1,50)

Once the data is loaded, reshape train and test data (X matrix)  to flattened vectors. 

A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:

### X_flatten = X.reshape(X.shape[0], -1).T           

Note : X.T is the transpose of X

For one example $x^{(i)}$:
$$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
$$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$ 
$$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$

The cost is then computed by summing over all training examples:
$$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$
