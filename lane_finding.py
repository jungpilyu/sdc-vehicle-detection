import numpy as np
import pickle
import cv2

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

def get_radius(y_eval, image_size, left_fit_cr, right_fit_cr):
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # first we calculate the intercept points at the bottom of our image
    left_intercept = left_fit_cr[0] * image_size[0] ** 2 + left_fit_cr[1] * image_size[0] + left_fit_cr[2]
    right_intercept = right_fit_cr[0] * image_size[0] ** 2 + right_fit_cr[1] * image_size[0] + right_fit_cr[2]
    calculated_center = (left_intercept + right_intercept) / 2.0
    lane_deviation = (calculated_center - image_size[1] / 2.0) * xm_per_pix
    return left_curverad, right_curverad, lane_deviation
    
def lane_fit(binary_warped2, nwindows=9, margin=100, minpix=50, visualization = False):
    """
    parameters
     - binary_warped :  warped binary image
     - nwindows : number of sliding windows
     - margin : width of the windows +/- margin
     - minpix : minimum number of pixels found to recenter window
     - visualization : drawing flag
    return
     - {left,right}_fit : fitting coefficients
    """
    # Assuming you have created a warped binary image called "binary_warped"
    binary_warped = np.copy(binary_warped2[:,:,0])
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if visualization:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Visualization of curve fitting
    # At this point, you're done! But here is how you can visualize the result as well:
    if visualization :
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, out_img, left_fitx, right_fitx
    
def reduce_noise(image, threshold=4):
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(image, ddepth=-1, kernel=k)
    image[nb_neighbours < threshold] = 0
    return image
    
def dir_threshold(img, sobel_kernel=3, thresh=(0, 0.2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1]+thresh[0])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobel = np.sqrt(sobelx*sobelx + sobely*sobely)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
def abs_sobel_thresh(img, orient='x', abs_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x' :
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    #    is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled)
    binary_output[(scaled >= abs_thresh[0]) & (scaled <= abs_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output
    
def hls_select(img, thresh=(0, 255)):
    img = np.copy(img)
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_ch = hls[:, :, 2]
    binary = np.zeros_like(s_ch)
    binary[(thresh[0] < s_ch) & (s_ch <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary
    
# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

def combine_all(image, color_thresh = (155, 255), gradx_thresh = (30, 255), grady_thresh = (25, 255), mag_thresh = (70, 255)): 

    # Apply each of the thresholding functions
    hls_binary = hls_select(image, thresh=color_thresh)
    gradx = abs_sobel_thresh(image, orient='x', abs_thresh=gradx_thresh)
    grady = abs_sobel_thresh(image, orient='y', abs_thresh=grady_thresh)
    mag_binary = mag_threshold(image, sobel_kernel=ksize, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.9, 0.1))

    combined = hls_binary | (gradx & grady) | mag_binary
    combined = 255 * np.dstack((combined, combined, combined)).astype('uint8')
    return combined
    
class birdsview:
    def __init__(self, before, after):
        """
        perspective transform wrapper class
        """
        self.M = cv2.getPerspectiveTransform(before, after)
        self.inverse_M = cv2.getPerspectiveTransform(after, before)

    def transform(self, image, direction = 'forward'):
        """
        cv2.warpPerspective() wrapper
        """
        size = (image.shape[1], image.shape[0])
        M = self.M if direction == 'forward' else self.inverse_M
        return cv2.warpPerspective(image, M, size, flags=cv2.INTER_LINEAR)
        
# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, cal_file =''):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None; self.bestx_left_buffer = []; self.bestx_right_buffer = []  
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        self.best_fit_left_buffer = []
        self.best_fit_right_buffer = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

        # choose a larger odd number to smooth gradient measurements
        global ksize
        ksize = 3 
        if cal_file == '':
            imgfiles = glob.glob('./camera_cal/calibration*.jpg')
            nx, ny = 9, 6
            cal_file = './cal_camera.p'
            calibrate_camera(imgfiles, nx, ny, cal_file)
         
        # get the camera calibration parameters, mtx, dist
        with open(cal_file, 'rb') as f:
            dist_pickle = pickle.load(file=f)
        self.mtx = dist_pickle['mtx']
        self.dist = dist_pickle['dist']
        
        # perspective transform
        before = np.array([[253, 697],[585,456],[700, 456],[1061,690]], np.int32)
        off = 60
        offset = np.array([[off, 0],[off, 0],[-off, 0],[-off, 0]], np.int32)
        after = np.array([[253, 697], [253, 0], [1061, 0], [1061, 690]] + offset , np.int32)

        src = np.float32(before)
        dst = np.float32(after)
        
        self.perspective = birdsview(src, dst)

    def update_fit(self, left_fitx, right_fitx, left_fit, right_fit):
        self.bestx_left_buffer.append(left_fitx)
        self.bestx_right_buffer.append(right_fitx)
        self.best_fit_left_buffer.append(left_fit)
        self.best_fit_right_buffer.append(right_fit)

        if len(self.bestx_left_buffer) >= 12:
            del self.bestx_left_buffer[0]
            del self.bestx_right_buffer[0]
            del self.best_fit_left_buffer[0]
            del self.best_fit_right_buffer[0]
            self.detected == True

    def get_best_fit(self):
        ave_left = np.average(self.best_fit_left_buffer, axis=0)
        ave_right = np.average(self.best_fit_right_buffer, axis=0)
        return ave_left, ave_right
    
    def get_bestx(self):
        ave_left = np.average(self.bestx_left_buffer, axis=0)
        ave_right = np.average(self.bestx_right_buffer, axis=0)
        return ave_left, ave_right
            
    def finding(self, img):
        undistorted_image = cv2.undistort(img, self.mtx, self.dist)
        binary_image = combine_all(undistorted_image)
        binary_image = reduce_noise(binary_image)
        # warped is a warped binary image
        warped = self.perspective.transform(binary_image)
        # left_fitx and right_fitx represent the x and y pixel values of the lines
        if self.detected == False:
            left_fit, right_fit, out_img, left_fitx, right_fitx = lane_fit(warped)
        else:
            left_fit, right_fit, out_img, left_fitx, right_fitx = fast_lane_fit(warped, self.best_fit[0], self.best_fit[1])
        self.update_fit(left_fitx, right_fitx, left_fit, right_fit)
        
        self.best_fit = self.get_best_fit()
        self.bestx = self.get_bestx()
        # ploty is arrays of fitting lines        
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        # curvature information and center offset, all in meters
        left_curva, right_curva, deviation = get_radius(np.max(ploty), warped.shape, self.best_fit[0], self.best_fit[1])
        self.radius_of_curvature = left_curva, right_curva
        self.line_base_pos = deviation
        curvature_text = 'Left Curvature: {:.2f} m    Right Curvature: {:.2f} m'.format(left_curva, right_curva)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undistorted_image, curvature_text, (100, 50), font, 1, (255, 255, 255), 2)
        deviation_info = 'Lane Deviation: {:.3f} m'.format(deviation)
        cv2.putText(undistorted_image, deviation_info, (100, 90), font, 1, (255, 255, 255), 2)
        
        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped).astype(np.uint8)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.bestx[0], ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.bestx[1], ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.perspective.transform(color_warp, 'inverse')
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
        return result

if __name__ == '__main__':
    from matplotlib.pyplot import plt
    import glob
    for image_p in glob.glob('test_images/test*.jpg'):
        line = Line('./cal_camera.p')
        image = mpimg.imread(image_p)
        draw_img = line.finding(image)

        fig = plt.figure(figsize=(15, 15))
        plt.imshow(window_img)
        plt.title('Raw Detections')
        plt.axis('off')
        fig.tight_layout()