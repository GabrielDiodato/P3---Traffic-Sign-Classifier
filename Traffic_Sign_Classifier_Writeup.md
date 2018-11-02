# Self-Driving Car Engineer Nanodegree

## Project: **Traffic Sign Classifier**

The goals / steps of this project are the following:

* Load the data set, explore, summarize and visualize the data set.
* Design, train and test a model architecture.
* Use the model to make predictions on new images.
* Analyze the softmax probabilities of the new images.
* Summarize the results with a written report.


### Reflection

### 1. Pipeline Description.

My pipeline consisted of 4 steps, as follows:

* Loading the Data Set, and Data Summary - it was used the numpy library to calculate summary statistics of the traffic signs data set.

* Model Architecture Development - the model architecture was based on the LeNet model architecture.

* New Images Testing -the model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

* Softmax Probabilities - were computed the top 5 softmax probabilities.


### 2. Potential Pipeline Shortcomings


I used a popular and simple LeNet CNN architecture. I see the biggest room for improvement here. Many modern Deep Learning systems use more recent and more complicated architectures like GoogLeNet or ResNet. This comes in more computational cost, on the other hand.


### 3. Pipeline Possible Improvements 

Perhaps augmenting the training set help to improve model performance. I would also like to investigate how alternative model architectures such as Inception, VGG, AlexNet, ResNet perfom on the given training set.
