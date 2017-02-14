# **Behavioral Cloning Project** 
---

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed
 
 The model is based on NVIDIA Self-Driving Cars pipeline as described in [this paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 
 
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

A dropout layer is added after the convolution layers and before the fully connected layer to reduce overfitting.

In order to prevent the model from learning the features of the environment(like the trees), the top section of the images was cropped. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

The images captured by the left, right and center camera were used for training the model. The training data largely consisted of driving in the center line.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I got inspiration for using the NVIDIA model from the [tips]((https://carnd-forums.udacity.com/questions/26214464/behavioral-cloning-cheatsheet) posted by Paul Heraty at the Udacity forums. Given the success of the model in driving an actual car further convinced me.

The original model, after implementation, performed very well on the training data set but had high MSE for the validation set, which implied that the model was overfitting.

To reduce the size of the model and improve accuracy the top and bottom sections of the image were cropped to prevent the model from learning features of the enviroment(like the trees) and remove the unnecessary section at the bottom of the image which does not contain any path markings. This improved the performance on the validation set.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

When I ran the model on the simulator I realised that it performed poorly, constantly veering off the road at the curves, not making sharp turns requires to stay on the track. Recording more data at the curves did not seem to help much, primarily because the keyboard control of the simulator was too sensitive, and in order to keep the car in the middle it required small quick key presses thus causing the simulator to record a 0 steering angle for most of the curve.

Since manually recording the curves was proving to be difficult I decided to use the left and the right camera image to train the model to steer back if it deviates from the center. This approach worked very well and the car was able to stay on the track, without being explicitly trained for recovery, which was difficult given the sensitive controls. A steering correction of 0.2 was found to be optimal.

In order to decrease the validatiion error further I introduced a dropout layer was added after the convolution layers and before the fully connected layer to reduce overfitting.

To make the training job faster, normalization and cropping was moved to the GPU while including it in the model's pipeline.

#### 2. Final Model Architecture

The final model architecture was largely the NVIDIA model except the dropout layer added after the convolutions. The NVIDIA model is show below for reference.

[NVIDIA Model](/nvidia_model.png?raw=true)

#### 3. Creation of the Training Set & Training Process

The data for the training was primarily obtained by driving in the center of the track. Recovery data recorded did not help the model and was discarded, given the limitations mentioned above.

The data for the model was augmented with the left and right camera images which made the model learn how to recover if the car is drifting to the side of the road.

The total recorded data consisted of 3479 log entries, resulting in 10437 images. The data was shuffled and split into an 80:20 ratio for the training and validation set respectively. 

Five epochs was find to be appropriate for training, since there was no signification improvement in accuracy using more epochs.
