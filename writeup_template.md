# **Project 3 - Behavioral Cloning**

## Project Definition

#### This Project is from the third part of Udacity's Self Driving Car Nanodegree Program and the goal is to make car drive autonomously in a simulation environment by providing only camera images. As an output steering wheel rotation will be determined by the model.
---

### Project Folder
TODO

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

1. In order to teach something to somebody, you have to show how it is done by providing examples, sometimes even more than once. Therefore in order to teach our car how to drive, in this project at least where & when to turn, first we drive the car manually on the given track and collect data from 3 camera images, left, center and right.

  ![3 Images left right center][image3]

2. As said sometimes there has to be multiple times of examples provided so that the given track is not memorised and thus can be applied to any kind of circumstances. Different kind of data collection tactics in order to generalise the model will be explained below. (TODO)

3. After I collected enough data I have created my own CNN Architecture by modifying the [NVIDIA Architecture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which is also used for same purposes but on a real car.

4. I have used Keras with a Tensorflow background in order to train my model.

5. Due to the high number of images, I have used [generator](https://wiki.python.org/moin/Generators) function of python while training my data, so I could only process necessary amount of data at a time. This functionality of Python allowed me to save the images in to RAM only when they had to be processed.

6. Processing of such a high number of images with this powerful network needs also computation power. My local computers CPU would handle this process in hours, therefore I decided to use E2C (Elastic Compute Cloud) provided by Amazon Web Services (AWS) to train my data in a GPU Instance.

7. Thanks to GPUs, my training process took around 1.5 hours and my model was ready to be test on autonomous mode of the simulator.


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

TODO - Cropping

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
