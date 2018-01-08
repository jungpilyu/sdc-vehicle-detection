## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image21]: ./writeup_images/2_1.png
[image221]: ./writeup_images/2_2_1.png
[image222]: ./writeup_images/2_2_2.png
[image223]: ./writeup_images/2_2_3.png
[image231]: ./writeup_images/2_3_1.png
[image232]: ./writeup_images/2_3_2.png
[image31]: ./writeup_images/3_1.png
[image32]: ./writeup_images/3_2.png
[image33]: ./writeup_images/3_3.png
[image34]: ./writeup_images/3_4.png
[image35]: ./writeup_images/3_5.png
[image361]: ./writeup_images/3_6_1.png
[image362]: ./writeup_images/3_6_2.png
[image363]: ./writeup_images/3_6_3.png
[image364]: ./writeup_images/3_6_4.png
[image365]: ./writeup_images/3_6_5.png
[image366]: ./writeup_images/3_6_6.png
[video1]: ./writeup_images/project_video_result.mp4
[video2]: ./writeup_images/project_video_result_with_lane.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

The files are submitted in the directory containing this write-up. The files are
* `CarND-Vehicle-Detection.ipynb` : a jupyter notebook which contains all the required codes.
* `CarND-Vehicle-Detection.html` : a html file exported by the jupyter notebook containing all the execution results.
* `./writeup_images/*` : all the images and video showing the result
*- `writeup_vehicle_detection.md` : this write-up file

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by examining the training dataset in png format provided for this project [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
The number of vehicle images is 8792 and that of non-vehicle images is 8968 which are well balanced. The png format in dataset has a shape of (64, 64, 3) and data type `float32` Some image samples are shown below.

![dataset][image21]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from the car images and displayed them to get a feel for what the `skimage.hog()` output looks like using HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

![HOG1][image221]
![HOG2][image222]
![HOG3][image223]

##### FEATURE EXTRACTION FROM DATA SET

Several feature extraction functions were defined. Here, three features are explored which are
- Raw pixel intensity which captures color and shape.
- Histogram of pixel intensity which captures color characteristics only.
- Gradients of pixel intensity which captures shape only.

Vertical image flipping can be used to augment the data set but I concluded that the data set without image flipping were enough to train a classifier. Finally, the MyClf class was defined which is a classifier class for the problem at hand.

#### 2. Explain how you settled on your final choice of HOG parameters.

Many parameters should be adjusted to get good performance. The type of parameters are summarized in the table below.

| Parameter     | Explanation | Examples |
|:-------------:|:-------------:|:-------------:|
| spatial_size | Spatial binning dimensions| (32,32) or (16,16) |
| hist_bins    | Number of histogram bins | 16, 32, 40, ...|
| color_space  | Color Space Conversion | 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb' |
| orient       | HOG orientation resolution | 9, 11, 13, ... |
| pix_per_cell | HOG Pixels per cell | 8, 12, ...|
| cell_per_block | HOG cells per block | 2, 3, ...|
| hog_channel  | channel to be applied in HOG | 0, 1, 2, or 'ALL' |

I tried various combinations of parameters but it was not easy to settle on the particular set because it was not evident to identify a specific good combination of parameters just by assessing test accuracy. Sometimes, I had to re-evaluate again by getting the final video annotation outcome. However this was very time-consuming iteration. So, I had to compromise between time and parameter optimization. Finally I settled on the following paramters

| Parameter    | Value |
|:------------:|:-------------:|
| spatial_size | (16,16) |
| color_space  | 'YUV' |
| orient       | 11 |
| pix_per_cell | 16 |
| cell_per_block | 2 |
| hog_channel  | 'ALL' |

I decided not to use the color channel histogram feature because it made many false positive in the final video outcome. So `hist_bins` parameter is undefined. The concatenated feature data is normalized by calling `StandardScaler` from `sklearn.preprocessing` library and shuffled and divided into training and test dataset in `train_test_split()` function.
The classifer training log is as follows.
```
51.34 Seconds to extract Cloor and HOG features...
A count of 8792  cars and 8968  non-cars of size:  1956 , 1956
Using spatial binning of: (16, 16) and 16 histogram bins
Using: 11 orientations 16 pixels per cell and 2 cells per block
Feature vector length: 1956
54.97 Seconds to train SVC...
Test Accuracy of SVC =  0.9961
My SVC predicts : [ 1.  0.  0.  0.  0.  1.  0.  1.  1.  1.  1.  0.  1.  0.  1.  1.  0.  0.]
For these labels: [ 1.  0.  0.  0.  0.  1.  0.  1.  1.  1.  1.  0.  1.  0.  1.  1.  0.  0.]
0.0625 Seconds to predict 18 labels with SVC
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

At first I trained a SVM with default arguments using my selected HOG and color features. However, there is an optimzation possibility in SVM training. So, I run the following code which try all the parameter combination
```python
parameters = {'kernel':('linear', 'rbf'), 'C':[0.8, 1, 2, 4, 8]}
X_train, X_test, y_train, y_test, X_scaler = get_vector(cars, notcars, **params)
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)
clf.fit(X_train, y_train)
print(clf.best_params_)
```


A classifier parameter tuning was done by `GridSearchCV` function from `sklearn.model_selection` library. The final parameters are the following.

| Parameter    | Value | Meaning |
|:------------:|:-------------:|:---:|
| kernel | (16,16) |Specifies the kernel type to be used in the algorithm.|
|  C | 4 | Penalty parameter C of the error term. |



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

