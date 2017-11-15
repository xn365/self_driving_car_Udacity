#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/1.png "Priority Road"
[image3]: ./examples/2.png "Stop"
[image4]: ./examples/3.png "No Entry"
[image5]: ./examples/4.png "No Passing"
[image6]: ./examples/5.png "Go straight or right"
[image7]: ./examples/6.png "Ahead Only"
[image8]: ./examples/German_Traffic_Signs.png "GTS"
[image9]: ./examples/augmented.png "Augmented"
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/xn365/self_driving_car_Udacity/tree/master/project2)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 * 32 * 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image8]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* First of all, I decided to use color images, because I think color is a useful feature.

* Then, due to the imbalanced number of samples of each class, I want to generate new data through data enhancement and improve the sample balance.

* The data enhancement methods used include: Random rotation, Random Translation, Random Transform and Random Brightness.

* After the data enhancement was completed, each class contained 5000 samples in the training data set.

* Because there isn't enough memory to put all the training data in,so I convert the training data to the TF.records file.

* Here is an example of an original image and an augmented image:
![alt text][image9]

* As a last step, I normalized the image data because it makes training easier

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
* Refer to the model of ResNet,I build small Resnet model.

* My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| 5 * Residual_Block_16 |                           outputs 16x16x16    |
| 5 * Residual_Block_32 |                           outputs 8 x 8x32    |
| 5 * Residual_Block_64 |                           outputs 4 x 4x64    |
| Global avg pool       |                           outputs 1 x 1x64    |
| Full Connect          |                           outputs 1 x 43      |
 

| Residual Block       	|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x16 or 16x16x16 or 8 x 8x32  			| 
| Batch Normalization   | 32x32x16 or 16x16x16 or 8 x 8x32              |
| Leak Relu             | 32x32x16 or 16x16x16 or 8 x 8x32              |
| Convolution1 3x3      | 1x1 stride, same padding                      |
| Batch Normalization   | 32x32x16 or 16x16x16 or 8 x 8x32              |
| Leak Relu             | 32x32x16 or 16x16x16 or 8 x 8x32              |
| Convolution2 3x3      | 1x1 stride, same padding                      |
| Convolution2 + Input  |       outputs 32x32x16 or 16x16x16 or 8 x 8x32|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adamOptimizer with learning rate 0.001.
The batch size is 128.
Number of steps is 3000.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.97~0.99
* validation set accuracy of 0.966 
* test set accuracy of 0.944

If a well known architecture was chosen:
* What architecture was chosen?
  I chose ResNet architecrue.
* Why did you believe it would be relevant to the traffic sign application?
  This is the best model I've ever heard of
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  The performance of the model is good, but the training time is not long enough to reach the optimum.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![alt text][image7]

The fifth and sixth images might be difficult to classify because they are very similar.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      	| Priority road   								| 
| Stop     			    | Stop 										    |
| No vehicles			| No vehicles						  			|
| No entry	      		| No entry					 				    |
| Turn left ahead		| Turn left ahead      							|
| Ahead only			| Ahead only      							    |

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.4%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

For the first image, the model is very sure that this is a priority road sign (probability of 0.99), and the image does contain a priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road   								| 
| .000002     			| Roundabout mandatory 							|
| .000002				| No entry										|
| .000001	      		| Turn left ahead					 			|
| .0000004			    | Children crossing      						|


For the second image, the model is very sure that this is a stop sign (probability of 0.97), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Stop   								        | 
| .023     			    | No passing 							        |
| .0019				    | Speed limit (20km/h)					        |
| .001	      		    | No passing for vehicles over 3.5 metric tons	|
| .0007			        | No entry      						        |

For the third image, the model is very sure that this is a no vehicles sign (probability of 0.99), and the image does contain a no vehicles sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .988         			| No vehicles   								| 
| .0088     			| No passing        							|
| .003				    | Keep left										|
| .000025	      		| End of all speed and passing limits			|
| .000016			    | Speed limit (70km/h)      					|

For the fourth image, the model is very sure that this is a no entry sign (probability of 0.91), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| No entry   								    | 
| .047     			    | No passing 							        |
| .021				    | Stop										    |
| .0057	      		    | No passing for vehicles over 3.5 metric tons	|
| .00038			    | Speed limit (20km/h)      					|

For the fifth image, the model is relatively sure that this is a turn left ahead sign (probability of 0.55), and the image does contain a turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .55         			| Turn left ahead  								| 
| .3     			    | Go straight or right 							|
| .1				    | Keep right									|
| .03	      		    | Ahead only					 			    |
| .006			        | Keep left      						        |

For the sixth image, the model is relatively sure that this is a ahead only sign (probability of 0.71), and the image does contain a ahead only sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .71         			| Ahead only   								    | 
| .22     			    | Turn left ahead 							    |
| .04				    | Go straight or right						    |
| .017	      		    | Keep right					 			    |
| .004			        | Keep left      						        |
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


