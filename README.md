#**Self-Driving Car Engineer Nanodegree** 

##Project-3: Behavioral Cloning

---

[image1]: ./images/raw_steering_distribution.jpg "Steering Angle Distribution before processing/augmentation"
[image2]: ./images/steering_angle_distribution.jpg "Steering Angle Distribution"
[image3]: ./images/nvidia_model.jpg "NVIDIA model"
[image4]: ./images/model_layers.jpg "Final model layers"
[image5]: ./images/iterations.jpg "Iterative steps"
[image6]: ./images/auto_mode_01.jpg "Auto-drive-01"
[image7]: ./images/auto_mode_02.jpg "Auto-drive-02"
[image8]: ./images/auto_mode_03.jpg "Auto-drive-03"


**Project Goals**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator.
* The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

---

### Files submitted (with this report)

* `model.py` - The script used to create and train the model, define generators, run deep neural networks for autonomous driving simulation
* `preprocessing.py` - helper function that performs various tasks like reading driving log data, image cropping and augmentation, etc.
* `drive.py` - Sends information to the simulator about throttle and steering angle.
* `model.h5` - saves model weights for future use
* `run.mp4` - video recording of vehicle driving autonomously for more than one lap

---

## Discussion & Overview

This project proved to be extremely frustrating and time consuming for me, to such an extent that I wanted to quit on several occassions. I spent more than three weeks trying to write and re-write the whole script, trying multiple DNN architechtures, data processing and augmentation methods, and finally it seems to be working. Detailes of the final approch can be found in the sections below, however, in a nutshell what I have done is as follows:

1. I recorded my own data using the simulator, in the process making sure that I have all scenarios captured, like sharp turns, driving out of track and coming back to the center, etc.
2. I started with Nvidia's model based on their paper `End to end learning for self-driving cars` and tried to add to the model to improve performance of my architechture.
3. I added regularization and dropout to prevent overfitting.
4. I added several data processing steps to speed up training and improve model prediction. Ultimately, after several tests I am using just a few of those. I have still kept some others in the `preprocessing.py` file, while deleted many others from it.
5. I played with several steering_correction parameters for left/right-side cameras, learning rate, epochs, data-sample split and PI control parameters in the `drive.py`. The final combination is what I am presenting in this submission


## 1. Data collection and processing

I recorded my own data using the simulator. In the process I made sure that I account for following driving conditions

#### 1.1 Data collection

* straight line driving
* sharp turns, in which I used to go to one edge of the lane and then turn suddently to get back in the center of the lane.
* drove a little out of the lane and came back to the center on multiple occassions.
* collected two sets, each over multiple laps. In first set I drove with heading direction being the default start direction in the simulator. In the second set I turned the car 180-degrees and drove multiple laps in the other direction. This was done to make sure that I have more or less a symmetric distribution of data about `0` steering angle.
* The data thus collected resulted in 2689 samples of images from a single camera with shape=(160, 320, 3). A plot of steering angle distribution w.r.t. central camera images is as shown in [image1]

![alt text][image1]

#### 1.2 Data processing

The data collected using the simulator needed some processing as suggested in lectures and online forums. Still I started testing with the raw data and soon realizeed that processing these large images took long time and the accuracy of trained model was not good. In autonomous mode my car kept falling off the track and mostly went under water :) 

For better memory management and improve training efficiency I tried several methods, some of which are:

* `3-camera-data`: I gathered data from all three cameras, offsetting steering values for left and right cameras by +/-(STEERING_CORRECTION)
* `horizontal-flip`: I added mirror images to the data set for all cases where steering angle was greater than 0.1. This was intentional to avoid `0` angle bias which a lot of people had reported in the forums (It took my a while to realize this).
* `random-shear`: I added random shear images to data set, as suggested by [Kaspar Sakmann in his article] (https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk)
* `cropping`: I cropped all images to cut top 70 and bottom 25 pixels, as these did not present significant lane information but mostly environmental details, that are not necessary for model training. And this cropping helps reduce memory occupied by the program.
* `re-sizing`: I resized all images to shape=(66, 200, 3), as was done in NVIDIA paper.
* `color-change`: I converted all images from BGR-to-RGB
* `others`: some other schemes like conversion from BGR-to-YUV, grayscaling, gamma-correction, etc were tried but not used in the final version. 

The augmented data set resulted in a total of 21,930 samples which was split into training-validation using 80-20 ratio. The steering angle distribution for this augmented data is shown in [image2].

![alt text][image2]



## 2. Model Architechture and training

#### 2.1 Model architechture

![alt text][image3]

I started training with the NVIDIA model described in their paper, as mentioned above and as shown in [image3]. This model was ok, but the convergence was slow and my car kept swaying off the road. Hence, I added some regularization, dropout and fully connected layers. After playing around with several combinations, I finally had an architechture with:

* Five 2D-convolution layers to increase feature depth (same as NVIDIA)
* I added non-linear `relu` activation after every convolution and fully connected layers. This was not part of NVIDIA architechture. I also tried `ELU` activation, but `relu` worked better.
* 5 fully connected layers, with final output being the predicted steering angle for the given image.
* 2 dropout layers to prevent overfitting.
* In addition to dropouts I have `L2-regularization` at every 2D-convolution and fully connected layers.
* All the layers are shown in [image4]

![alt text][image4]

#### 2.2 Training

For training the model, I perform a 80-20, training-validation sample-data split and the optimization is performed using Adam optimizer with 0.0002 as learning rate, mean-squared-error minimized with a batch size of 128, iterated over 10 epochs.

`STEERING_CORRECTION` is one of the parameters that needs adjustment for successful training of this model. I tried various values between 0.2 and 0.4 and finally felt that 0.25 was the best for which I did not have to make any steering or throttle control adjustment in the `drive.py`. Hence, I am using `drive.py` as is except for processing the image that is being fed to steering prediction model.


## 3. Autonomous driving with the model

The final model-weights were saved in `model.h5`. I used the following command along with the simulator to run it in autonomous mode

`>> python drive.py model.h5 run`

The resulting images were saved in folder titled `run` and were used to generate a `run.mp4` video. The driving in autonomous mode is mostly very smooth. During an entire lap the car sways to the edge of the lane on 3-4 occassions, but quickly checks itself and comes back to the center of the lane. The link to video is [here](./run.mp4)

Some sample snapshots for the track are also shown here in [image6] [image7] and [image8]

![alt text][image6]
![alt text][image7]
![alt text][image8]
 




