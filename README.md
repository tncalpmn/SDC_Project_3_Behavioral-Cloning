# **Project 3 - Behavioral Cloning**

## Project Definition

#### This Project is from the third part of Udacity's Self Driving Car Nanodegree Program and the goal is to make car drive autonomously in a simulation environment by providing only camera images. As an output steering wheel rotation needed to be determined by the model.
---

### Project Folder

Short overview about which file contains what:
* drive.py -> Python script to load trained model and to connect it with simulators autonomous mode.
* model.h5 -> My trained Keras model that works perfectly on both test track and challange track. (Note that model is trained with Keras 2.1.6 version, therefore drive.py has to be run on the same environment that has same Keras version installed)
* Track_1 & 2.mp4 -> Video of the recorded images during testing the model on test and challange track. (Note that Links to birds eye view videos with better quality can be found at the end of this document)
* model.py -> Contains the code & network to train images. (Note that due to the big size of the dataset, training images are not provided in this repo. However, below is explained how the training images are collected)

---

### Aftermath
Here is a list of API functions and python features that I have been using along this project (as a future reference to myself).

* **PIL.Image.transpose** -> is used to flip an image around a given axes. ex:PIL.Image.FLIP_LEFT_RIGHT [Doc](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)
* **with open(path,'w') as csv_file** -> try/finally block in order to open a file safely
* **csv.writer(csv_file, delimiter=',')** -> prepares csv template to write
* **writer.writerow(line)** -> actually writes in to the template (each line can be written with the help of a loop)
* **csv.reader(path)** -> reads the data from a given path (each line can be read with the help of a loop)
* **PIL.Image.crop*** -> crops the image by given dimensions: as input coordinates of cropped pixel(left, top, right, bottom)
* **Sequential()** -> Creates an empty sequential model in Keras, which can be filled by various networks.
* **Cropping2D(cropping=((top,bottom),(left,right)))** -> crops an image inside of a Keras without actually changing the dimensions. Advantage of this method is that during the prediction stage, original dimensions of image can be kept when providing it to model.
* **Lambda(lambda x: do Smt to x)** -> to each input applies the given formula (can be used as normalisation of the image)
* **Convolution2D(noOfFilter,(x,y),strides=(x, y))** -> applies convolutional neural network: as parameter  takes number of filter, size of filter and size of shifting (strides), also how is shifting done via 'Same' or 'Valid' (padding) default is "Valid" in Keras.
* **Dropout(procent)** -> value between 0 - 1 indicated what percent of the parameter has to be cut off -> learning less but more general. Prevents overfitting
* **Activation()** -> Derivable activation functions that actually does the work. Some common ones are 'elu', 'relu', 'softmax', 'tanh', 'sigmoid' -> provides nonlinearity
* **Flatten()** -> Flattens the data into one dimention: 12x12*3 = 432 x 1
* **Dense(noOfFilter)** -> Simply applies fully connected layer
* **model.compile(loss='',optimizer='')** -> Compilation paramaters as loss function and type of optimizer to be used
* **model.fit_generator(data, steps_per_epoch, valData, num_validation_sample, epochs, verbose=1)** -> Train the data by actual values. Returns loss and val_loss parameters per epoch
* **model.save(model.h5)** -> Saves trained Keras model locally
* **model.load_model(path)** -> Loads Keras model
---

### Framework

1. In order to teach something to somebody, one has to show how it is done by providing examples, sometimes even more than once. Therefore in order to teach the car how to drive, at least where & when to turn for the scope of this project, first I drive the car manually on the given track and collect data from 3 camera images from left, center and right.

  ![3 Images left right center][image1]

2. As said sometimes there has to be multiple times of examples provided so that the given track is not memorised and thus can be applied to any kind of circumstances. Different kind of data collection tactics in order to generalise the model will be explained below.

3. After I collected enough data I have created my own CNN Architecture by modifying the [NVIDIA Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which was also used for same purposes but on a real car.

4. I have used Keras with a Tensorflow background in order to train my model.

5. Due to the high number of images, I have used [generator](https://wiki.python.org/moin/Generators) function of python while training my data, so I could only process necessary amount of data at a time. This functionality of Python allowed me to save the images in to RAM only when they had to be processed.

6. Processing of such a high number of images with this powerful network needs also computation power. My local computers CPU would handle this process in hours, therefore I decided to use E2C (Elastic Compute Cloud) provided by Amazon Web Services (AWS) to train my data in a GPU Instance.

7. Thanks to GPUs, my training process took around 1.5 hours and my model was ready to be test on autonomous mode of the simulator.

Type the following to terminal in order to connect to simulators autonomous mode:
~~~~
python drive.py model.h5
~~~~


[//]: # (Image References)

[image1]: ./WriteUpImages/three_images.png "3 Images"
[image2]: ./WriteUpImages/nvidia_arc.png "Nvidia"
[image3]: ./WriteUpImages/elu-and-relu.png "Elu & Relu"
[image4]: ./WriteUpImages/mse.png "MSE"
[image5]: ./WriteUpImages/cropped.png "RecovCroppedery Image"
[image6]: ./WriteUpImages/Flipped.jpg "Normal Image"
[image7]: ./WriteUpImages/Original.jpg "Flipped Image"


---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Here are the layers of my modified NVIDIA network (model.py lines 58-116):



| Layer         		|     Description	        					| Stride|Filter Size| Padding |
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x323x3 RGB image   							| - | - | - |
| Cropping         		| Crop Image 66 px from top and 22 px from bottom   							| - | - | - |
| Lambda    	| Normalise Inputs (pixels) by (val/127.5)-1 	|- | -|-|
| Convolution     	| Output depth 24 	| 2x2|5x5 |Valid|
| RELU					|						-						| -| -|-|
| Dropout					|	drop	0.5 during training 										| -| -|-|
| Convolution     	| Output depth 36 	| 2x2|5x5 |Valid|
| RELU					|						-						| -| -|-|
| Dropout					|	drop	0.5 during training 										| -| -|-|
| Convolution     	| Output depth 48	| 2x2|5x5 |Valid|
| RELU					|				-								|- |- |-|
| Dropout					|	drop	0.5 during training 										| -| -|-|
| Convolution     	| Output depth 64 	| 3x3|1x1 |Valid|
| RELU					|						-						| -| -|-|
| Dropout					|	drop	0.5 during training 										| -| -|-|
| Convolution     	| Output depth 64	| 3x3|1x1 |Valid|
| RELU					|			-									| |- |-|
| Flatten      	|-		| -| -|-|
| Fully connected		| outputs 100        									|- |- |-|
| Dropout					|	drop	0.5 during training 										| -| -|-|
| ELU					|				-								| -|- |-|
| Fully connected		| outputs 50        									|- |- |-|
| Dropout					|	drop	0.5 during training 										| -| -|-|
| ELU					|				-								| -|- |-|
| Fully connected		| outputs 10        									|- |- |-|
| ELU					|				-								| -|- |-|
| Fully connected		| outputs 1        									|- |- |-|


This is inspired by following [architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/):

  ![NVIDIA Network][image2]



For sake of trying a different activation function in this project, I have used ELU activation function instead of RELU which is very similar to RELU but also accepts negative inputs.

![EluAndRelu][image3]

[Image Ref](https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/)

Although, it is hard to say/compare the performance of both activations functions on this task, I was very satisfied with the performance of my model.

#### 2. Attempts to reduce overfitting in the model

Before I train my images on a CPU, I wanted to try my code on couple images in my local computer. Switch limitDataSample = -1 at line 20 in model.py limits the amount of data given to network. I realised that the size of my output model.h5 after giving only 20 Images to network was bigger than 1.5 GB. This was simply because my network was missing Dropout and Stride functionalities. When these not defined, model was trying to save all parameters in to the output model. That would probably end up overfitting and causes validation accuracy to be low. After I set my "filtering" layers, my model size was not higher than 10Mb. The loss of training and validation sets after each epoch was low and similar (~0.025), which is a sign of not over/underfitting.

As a test set here I used the simulators autonomous mode in order to check the accuracy. Since this is a regression-> Dense(1) task, it is harder to evaluate a test with numbers. "Is car staying on track without driving over lanes?" was however achieved.

#### 3. Model parameter tuning

I have compiled my model with adam optimiser, with what I have got a good performance already from the last project. As mentioned before, since this is a regression task, instead of softmax, **mean squared error** function is used to estimate the total error.

![MSE formula][image4]

[Image Ref](https://en.wikipedia.org/wiki/Mean_squared_error)

Additionally..
- Number of epoch was selected as 2
- Batch Size 32
- Number of Training Images: 32288 & Validation Images: 8072

#### 4. Appropriate training data

Simulator allows images to be recorded as we drive in training mode, and therefore a dataset creation was possible. I have created 7 Samples with following driving behaviours:

1. Driving car **1** lap keeping the car always in the middle
2. Driving car **2** laps fast.
3. I have used sample dataset that was provided by Udacity
4. Driving car in reverse-track x **1** slow (clock-wise)
5. Driving car in reverse-track x **2** fast (clock-wise)
6. Recovery track, I have teach the car to strongly turn left or right when it was close to side lanes. Such kind of driving look liked quite drunken though.
7. To generalise the training I have also captured one full lap from challenge track. (Which probably made possible car to drive in challenge track very good too)

In addition to this data I have generated flipped images from each samples and then added them in to training set. (Note that when image is flipped then steering had to be edited by multiplying -1). All and all I had over 40.000 images just from central camera. I have not used left and right images since I was satisfied with the performance of the model.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I have benefited NVIDIAs Network which was also used for same purposes but on a real vehicle, and as expected worked very well on the simulation too.

Despite this networks powerful characteristic, the amount of  training data was decisive on the success. I had to train tree different datasets in order to achieve best driving without any mistakes.

In order to get a numerical evaluation, I have split the data by 20 procent as evaluation and let model compare the predicted steering with ground truth, which is actually my input with keyboard to the system.

Besides low and similar loss values on training and validation data, I have proved that the model is not overfitting the training set by simply applying my model to challenge track. On first try model was able to finish this hard track without any mistakes.

I have to note that though running same model multiple times on autonomous model gave different driving behaviours.


#### 2. Creation of the Training Set & Training Process

In order to reduce the training time and for sake of feeding network with only useful information, I have cropped the image from top and bottom, so that network would focus on the features of the road. Cropping is done in Keras in place before fed into convolutional layer, however here is a cropped image I have done manual for visualisation purposes, from top: 64px and bottom 22px was cropped:

![Cropped Image][image5]

Flipping as a data augmentation process:

Original versus flipped Image:

![alt text][image6]
![alt text][image7]

At the end my model was successful enough to drive safely on both tracks. Here are the links [Track 1](https://youtu.be/P4RXyGH9lbI) & [Track 2](https://youtu.be/yIgM4hO3RfM) to the videos that I have captured during autonomous mode.
