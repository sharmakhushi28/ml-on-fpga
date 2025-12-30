a) Write your own convolution Y=f*g and correlation Y=f•g algorithms. Use a 7X7 matrix for f, with all zeros except
the center value, which should be 1. For g use a 3X3 matrix with values [[1,2,3],[3,4,5],[6,7,8]]. Write a 2D
convolution to convolve the 7X7 and 3X3. Use valid padding. Correlate and the convolve f with g and show the
results. Submit your code, your results and answer to this question. (stride = 1 for this problem)
A square reference correlation function and convolution function are shown below for reference so that you
can check your work.
import scipy.signal as ss
def conv2Dd_(image,W,stride,Conv):
if (Conv):
y = ss.convolve2d(image, W, mode='valid') ## valid padding
else:
y = ss.correlate2d(image, W, mode='valid') ## valid padding
Xdim = len(image[0])//stride
x = np.zeros([Xdim,Xdim],float)
if stride>1: ## implement stride
for i in range(0,Xdim):
for j in range(0,Xdim):
x[i,j] = y[i*stride,j*stride]
else:
x = y
return(x)
This is a simple, square matrix scipy 2d convolution that you can use to check your algorithm
note: that this algorithm may have some weird scaling (1/sqrt(2pi) would have made sense, but it isn’t quite that), a
scalar multiplied by the result, but that shouldn’t matter to the result, the scaling should be consistent over the
elements of the result
In fact, a convolution is a correlation with a 180 degree rotated weight matrix. To avoid doing this rotation and to make
the results more straight forward the correlation is often used instead of convolution. If you rotate the weight matrix
and use correlation you are effectively doing a convolution which is what we are doing here.
So, they could be used in 2D only by flattening the portion of the matrix overlapping the weights, and the weights on each step

b) In this problem you will write the forward propagation for a CNN followed by a simple ANN which will take its outputs and turn them into classifications. Using the 2D convolution algorithm you wrote above do forward propagation, or prediction based on these given weights. (stride = 2 for this problem)
Try the convolution weights
W1 = np. array([[ 1.6975548, -0.07326141, -0.41880725],
[ 0.12228276, -0.19572004, 0.81986898],
[ 0.8876136, -1.8629187, -0.97661273]])
## And the ANN weights
W2 = np. array([ 1.10485759, 0.2120758, -1.31693339])
Don’t forget to standardize each image (shown below) so that the mean is 0 and the standard deviation is 1. If you don’t do this the weights I have given you may not work.
To get X3 matrix multiply the output of the correlation/convolution function W2. Use a quantizer which sets the maximum prediction to 1 and the others to 0 for each image. Maybe like this:
##
## categorize
##
def categorize(X):
XX = np.zeros(len(X),float)
maxind = X.tolist().index(max(X.tolist()))
XX[maxind] = 1.0
return(XX)
Predict each of the “images” and compare to the targets. The algorithm should classify your training and test cases correctly, but you may have to decide whether you should rotate the weights or not.
Turn in your code, your results, organized so that the grader can find all your code data and results for each problem. If they can’t find something they will assume 0.
Be careful copying code from office products to Spyder or other editors, some characters are changed like single quotes. Sometimes invisible characters occur causing you to get an error on a line until you completely retype it.
import numpy as np
import matplotlib.pyplot as plt
Image_dim = 7
stride = 2
Ydim = 3
Wdim = 3
##
## target
##
target = np.array([[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],[0,0,1]],float)
##
## training data
##
## 0
## this is a form that is easier to work with, the other form works too
image7by7 = np.zeros([9,Image_dim,Image_dim],float)
image7by7[0,1,:] = np.array([0,0,0,1,0,0,0])
image7by7[0,2,:] = np.array([0,0,1,1,0,0,0])
image7by7[0,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[0,4,:] = np.array([0,0,0,1,0,0,0])
image7by7[0,5,:] = np.array([0,0,1,1,1,0,0])
## 1
image7by7[1,1,:] = np.array([0,0,1,1,0,0,0])
image7by7[1,2,:] = np.array([0,0,0,1,0,0,0])
image7by7[1,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[1,4,:] = np.array([0,0,0,1,0,0,0])
image7by7[1,5,:] = np.array([0,0,0,1,0,0,0])
## 2
image7by7[2,1,:] = np.array([0,0,0,1,0,0,0])
image7by7[2,2,:] = np.array([0,0,0,1,0,0,0])
image7by7[2,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[2,4,:] = np.array([0,0,0,1,0,0,0])
image7by7[2,5,:] = np.array([0,1,1,1,1,1,0])
## 3
image7by7[3,1,:] = np.array([0,1,1,1,1,1,0])
image7by7[3,2,:] = np.array([0,1,0,0,1,0,0])
image7by7[3,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[3,4,:] = np.array([0,1,1,0,0,0,0])
image7by7[3,5,:] = np.array([0,1,1,1,1,1,0])
## 4
image7by7[4,1,:] = np.array([0,0,1,1,1,0,0])
image7by7[4,2,:] = np.array([0,0,0,0,1,0,0])
image7by7[4,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[4,4,:] = np.array([0,0,1,0,0,0,0])
image7by7[4,5,:] = np.array([0,0,1,1,1,0,0])
## 5
image7by7[5,1,:] = np.array([0,0,1,1,0,0,0])
image7by7[5,2,:] = np.array([0,1,0,0,1,0,0])
image7by7[5,3,:] = np.array([0,0,0,1,0,0,0])
image7by7[5,4,:] = np.array([0,0,1,0,0,0,0])
image7by7[5,5,:] = np.array([0,1,1,1,1,1,0])
## 6
image7by7[6,1,:] = np.array([0,1,1,1,1,1,0])
image7by7[6,2,:] = np.array([0,1,0,0,0,1,0])
image7by7[6,3,:] = np.array([0,0,1,1,1,1,0])
image7by7[6,4,:] = np.array([0,0,0,0,0,1,0])
image7by7[6,5,:] = np.array([0,1,1,1,1,1,0])
## 7
image7by7[7,1,:] = np.array([0,0,1,1,1,0,0])
image7by7[7,2,:] = np.array([0,1,0,0,0,1,0])
image7by7[7,3,:] = np.array([0,0,0,1,1,0,0])
image7by7[7,4,:] = np.array([0,1,0,0,0,1,0])
image7by7[7,5,:] = np.array([0,0,1,1,1,0,0])
## 8
image7by7[8,1,:] = np.array([0,1,1,1,1,1,0])
image7by7[8,2,:] = np.array([0,1,0,0,0,1,0])
image7by7[8,3,:] = np.array([0,0,0,1,1,1,0])
image7by7[8,4,:] = np.array([0,1,0,0,0,1,0])
image7by7[8,5,:] = np.array([0,1,1,1,1,1,0])
## print('image 1 \n', image7by7)
##
## test data
##
targett = np.array([[1,0,0],[0,0,1],[0,1,0]],float)
##
image7by7t = np.zeros([3,7,7],float)
image7by7t[0,1,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,2,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,3,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,4,:] = np.array([0,0,0,1,0,0,0])
image7by7t[0,5,:] = np.array([0,0,0,1,1,0,0])
## print('image 1 \n', image7by7t0)
image7by7t[1,1,:] = np.array([0,0,1,1,1,0,0])
image7by7t[1,2,:] = np.array([0,1,0,0,0,1,0])
image7by7t[1,3,:] = np.array([0,0,0,1,1,1,0])
image7by7t[1,4,:] = np.array([0,1,0,0,0,1,0])
image7by7t[1,5,:] = np.array([0,0,1,1,1,0,0])
## print('image 1 \n', image7by7t1)
## test image a 2
image7by7t[2,1,:] = np.array([0,0,1,1,1,1,0])
image7by7t[2,2,:] = np.array([0,1,0,0,1,0,0])
image7by7t[2,3,:] = np.array([0,0,0,1,0,0,0])
image7by7t[2,4,:] = np.array([0,0,1,0,0,0,0])
image7by7t[2,5,:] = np.array([0,1,1,1,1,1,0])
Also one way to 180 degree rotate a matrix
##
## reverse order columns and then rows
##
def M180deg(M):
return(np.flip(np.flip(M,axis=1),axis=0))

c) Copy this algorithm to a different file and add back propagation to determine the weights to classify these images into class 1, 2 and 3. Start with random weights, I used these
##
## training
## initial settings
##
np.random.seed(1)
W1 = (np.random.rand(Wdim,Wdim)-0.5)
print(W1)
W2 = (np.random.rand(Wdim)-0.5)
print(W2)
(stride = 2 for this problem, must be consistent with problem 2, right?)
You will need to write a function to find dzdW1 and a function to find dzdW2. Show your algorithms clearly. You can check your result using autograd but write your own function.
You should get the weights shown in 2, but your weights may be scaled differently.
Turn in the code and the results of your run. Make which weights you found clear. Apply the weights you found to the training data and the test dataset and show how well you did in matching each.