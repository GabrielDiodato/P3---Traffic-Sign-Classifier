# Self-Driving Car Engineer Nanodegree

## Project: **Traffic Sign Classifier**

The goals / steps of this project are the following:

* Load the data set, explore, summarize and visualize the data set.
* Design, train and test a model architecture.
* Use the model to make predictions on new images.
* Analyze the softmax probabilities of the new images.
* Summarize the results with a written report.

[//]: # (Image References)

[bar_chart_training_set]: ./figures/bar_chart_training_set.png "Distribution of training samples per label"
[labels_with_examples]: ./figures/labels_with_examples.png "Labels and example images"
[grayscale]: ./figures/grayscale.jpg "Grayscaling"
[traffic_signs_orig]: ./figures/traffic_signs_orig.png "Traffic Signs"
[traffic_signs_prediction]: ./figures/traffic_signs_prediction.png "Traffic Signs Prediction"
[learning]: ./figures/learning.png "Validation Accuracy per Epoche"
[prediction_probabilities_with_examples]: ./figures/prediction_probabilities_with_examples.png "Traffic Sign Prediction with Examples"
[prediction_probabilities_with_barcharts]: ./figures/prediction_probabilities_with_barcharts.png "Traffic Sign Prediction with Bar Charts"
[model_architecture]: ./figures/model_architecture.png "Architecture of Model"

### Reflection

### 1. Pipeline Description.

My pipeline consisted of 4 steps, as follows:

#### Loading the Data Set, and Data Summary - it was used the numpy library to calculate summary statistics of the traffic signs data set.

The data set summary is presented below:

* The size of training set is  34799.
* The size of the validation set is  4410.
* The size of test set is  12630.
* The shape of a traffic sign image is  (32, 32, 3).
* The number of unique classes/labels in the data set is = 43.

The figure below shows one example image for each label in the training set.

![alt text][labels_with_examples]

The following figure presents the exploratory visualization of the data set. The chart shows how many samples are contained in the training set per label.

![alt text][bar_chart_training_set]

#### Model Architecture Development - the model architecture was based on the LeNet model architecture.

* Two preprocessing technniques were used, image grayscaling (used because several images in the training were quite dark and contained only little color und the grayscaling reduces the amount of features and thus reduces execution time) and image normalizing (used to convert the int values of each pixel [0,255] to float values with range [-1,1]).

* The model architecture was based on the LeNet model architecture. Dropout layers were also added before each fully connected layer in order to avoid overfitting. The model consisted of the following layers:

| Layer                  |                 Description                    |
|------------------------|------------------------------------------------|
| Input                  | 32x32x1 gray scale image                       |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 14x14x6                   |
| Convolution 5x5        | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU                   |                                                |
| Max pooling            | 2x2 stride,  outputs 5x5x16                    |
| Flatten                | outputs 400                                    |
| **Dropout**            |                                                |
| Fully connected        | outputs 120                                    |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 84                                     |
| RELU                   |                                                |
| **Dropout**            |                                                |
| Fully connected        | outputs 43                                     |
| Softmax                |                                                |

* To train the model, an Adam optimizer was run and the following hyperparameters were set:
Batch size: 128.
Number of epochs: 150.
Learning rate: 0.0006.
Variables were initialized by truncated normal distribution with mu = 0.0 and sigma = 0.1.
Keep probalbility of the dropout layer: 0.5.

* An iterative approach has been used to optimize the validation accuracy:
The original LeNet model from the online course was chosen. To adapt the architecture for the traffic sign classifier, the input was changed in order to accept the images from the training set with shape (32,32,3) and the number of outputs were set to mstch the 43 unique labels in the training set. The initial training accuracy was **83.5%** and the test traffic sign "pedestrians" was not correctly classified. (The initially used hyperparameters were: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1). 

After adding the grayscaling preprocessing, image normalization, tuning hyperparameters and adding dropout layers, it was possible to achieve a validation accuracy equals to **97,5%**. (The final used hyperparameters were: EPOCHS = 150, BATCH_SIZE = 128, rate = 0,0006, mu = 0, sigma = 0.1).

The figure below show how the validation accuracy varies as a function of the number of EPOCHS for the final model:
   
![alt text][Learning]

#### New Images Testing - the model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

* 6 German traffic signs were found on the web and are presented below:

![alt text][traffic_signs_orig]

* The results of the prediction are presented below:

![alt text][traffic_signs_prediction]

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.1%.

#### Softmax Probabilities - were computed the top 5 softmax probabilities.

The following figures present the top five softmax probabilities of the predictions on the captured images are outputted. As shown in the bar chart the softmax predictions for the correct top 1 prediction is bigger than 98%. 

![alt text][prediction_probabilities_with_barcharts]

The detailed probabilities and examples of the top five softmax predictions are given below.

![alt text][prediction_probabilities_with_examples]


### 2. Potential Pipeline Shortcomings


I used a popular and simple LeNet CNN architecture. I see the biggest room for improvement here. Many modern Deep Learning systems use more recent and more complicated architectures like GoogLeNet or ResNet. This comes in more computational cost, on the other hand.


### 3. Pipeline Possible Improvements 

Perhaps augmenting the training set help to improve model performance. I would also like to investigate how alternative model architectures such as Inception, VGG, AlexNet, ResNet perfom on the given training set.
