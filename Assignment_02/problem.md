a) Write your own Perceptron code. Train using the dataset using the data given below. Inputs are columns x0,x1,x2 which have only 5 observations and output
target known is y. Use the Perceptron cost function and mathematics from the perceptron class lecture. Use compact notation.
Run it for the dataset provided and turn screen grabs of your output. An iteration is a complete pass through all the data. This is an iterative solution so it will have multiple passes, but you want it to have as few as possible. A quantizer is used on the output, all positives become 1, all negatives become -1.
Your output should contain
1) A display of the value of the prediction, weights, and error in each iteration as the algorithm converges to a solution
2) At the end a plot of error between your prediction of y and the actual y (which we call a target) as a function of iteration
3) Percent error
(I used from sklearn.metrics import accuracy_score print('Accuracy: %.2f' % accuracy_score(y, ypreds))
)
4) Number of misclassified cases in the final prediction
Make sure to turn in your code files and screen captures of all output.

b) Adaline was a competing machine learning algorithm shortly after the
Perceptron was published. Think of Adaline as a Neural Network with three
major differences with the Perceptron. First it has an L2 Norm cost function.
Second it has a different learning rule. The Perceptron learning rule is that it
only updates weights when there is an error. In this case the learning rule is to
always move the prediction closer to the classification by updating the weights
even when the quantized classification is correct. The third difference is the
activation function is linear as in it doesn’t have one contrasted with the
Perceptron ReLU activation function.
As a consequence Adaline requires more mathematical resources. It uses
gradient descents. Like the Perceptron, it has a quantizer to translate the
linear output into 1 and -1, which is explicitly shown in the flow here. Use the
least-squares cost function for your algorithm. Study the Adaline flow shown
before and try to understand the differences between it and the Perceptron
flow.
1) Use a random 2/3 of the dataset for training and 1/3 for testing as
described in the lecture
2) Normalize/standardize the dataset as described in the lecture
3) If you do update the weight for all observations, rather than the ones that have error your solution may find the best solution and the diverge away from that solution so stop iterating when the error drops below a tolerance, I used while (t<epochs and total_error/Nobs>tol): where Nobs is the number of observations, and tol=0.001.
In your assessment of whether it has error, you should quantize your prediction.
Also, we need a larger dataset to do this because we average weights over a statistical sampling. Two are provided. Dataset_1.csv has no noise, and Dataset_2.csv has noise.
Develop you code with Dataset_1.csv and then run it again on Dataset_2.csv. Comment on your results. Results in each case should contain %error, number of misclassification, and number of iterations.
A pseudo code for Adaline
t = 0; epochs = 100; tol = 1e-6;
while (t<epochs and abs(total_error-last_total_error)>tol):
last_total_error = total_error
Ypred = model(X,w)
Error = Ypred-Y
w -= eta*dCostF(Error,X)
total_error = np.dot(Error,Error)
errors.append(total_error/len(X))
t += 1
Yq = quantizer(Ypred)
print( t, total_error,Yq)
plot total_error vs t
dCostF is the derivative of the cost function. There are of course many ways to implement it but this is perhaps the simplest. This is just a guide.
Use something like sklearn.linear_model.Perceptron to compare your results for this and problem 4. This is a canned library version of the Perceptron algorithm which makes a nice reference. Screen capture your results. Compare number of miss-classifications, and accuracy for each dataset. As always, hand in your code file.