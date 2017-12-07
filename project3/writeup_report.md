# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_1.jpg "Center Image"
[image1]: ./images/left_1.jpg "Left Image"
[image1]: ./images/right_1.jpg "Right Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 video of the car in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model is defined in lines 94-123 of `model.py`.

My model consists of a convolution neural network with 5x5 or 3x3 filter sizes and depths between 16 and 64 (model.py lines 103-107) 

The data is normalized in the model using a Keras lambda layer (code line 101).


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 109, 112, 115 and 118). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 127-132). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the most powerful model I know.

My first step was to use a convolution neural network model similar to the ResNet I thought this model might be appropriate because it is so powerful.

But when I tried that model (50layers), I found it needed too much resources, It takes 1 minutes to train one step on my computer.

So, I go back and use a simple one.

#### 2. Final Model Architecture

The final model architecture (model.py lines 94-123) consisted of a convolution neural network with the following layers and layer sizes:

The layers are summarized in the table below:

|Layer (type)           |          Output Shape     |
| Normalization (Lambda)|      (None, 66, 200, 3)   |
| conv2d_1 (Conv2D)     |      (None, 31, 98, 16)   |
| conv2d_2 (Conv2D)     |      (None, 14, 47, 32)   |
| conv2d_3 (Conv2D)     |      (None, 5, 22, 64)    |
| conv2d_4 (Conv2D)     |      (None, 3, 20, 64)    |
| conv2d_5 (Conv2D)     |      (None, 1, 18, 64)    |
| dropout_1 (Dropout)   |      (None, 1, 18, 64)    |
| flatten_1 (Flatten)   |      (None, 1152)         |
| dense_1 (Dense)       |      (None, 128)          |
| dense_2 (Dense)       |      (None, 64)           |
| dense_3 (Dense)       |      (None, 32)           |
| dense_4 (Dense)       |      (None, 1)            | 


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

The images are cropped in order to remove unnecessary interference.

For left camera image, steering angle is adjusted by +0.25.

For right camera image, steering angle is adjusted by -0.25.

Randomly flip image left/right with 50% probability.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the valid loss did not decrease anymore. I used an adam optimizer so that manually training the learning rate wasn't necessary.
