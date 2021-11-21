''' ............................................................................

The Unit Test for Master thesis:
"Development of a lane keeping algorithm based on computer vision for an autonomous car model"

Behrooz Bonakdar Yazdi
August 2020

.............................................................................'''


from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


### ........................................................................................................................... ###

# Defining a Function for getting the gradiant image by canny method:
def canny(image):   #this image is different with image downside. here it is the argument of function
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #make the copied image gray scale
    blur = cv2.GaussianBlur(gray,(5, 5),0) #smoothening the image by using a guassian filter
    canny = cv2.Canny(blur,50,150) #find the gradient image by canny method
    return canny

### ........................................................................................................................... ###
### ROI:
# Defining the near field region of intrest (ROI):
def near_ROI(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([            # defining the dimensions for the region of interest
    [(0, height),(1000, height),(1000-100,450),(100, 450)]
    ])
    mask = np.zeros_like(image)  # make an black image with dimensions of the input image
    cv2.fillPoly(mask, polygons, 250) # apply white(250) on mask with dimensions defined on polygons
    near_masked_image = cv2.bitwise_and(image,mask) #with bitwise AND we choose only lines in Region of Interest
    return near_masked_image

# Defining the far field region of intrest (ROI):
def far_ROI(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([            # defining the dimensions for the region of interest
    [(350, 720),(470, 720),(470,370),(350, 370)] #[(300, 450),(900, 450),(700,350),(500, 350)]
    ])
    mask = np.zeros_like(image)  # make an black image with dimensions of the input image
    cv2.fillPoly(mask, polygons, 250) # apply white(250) on mask with dimensions defined on polygons
    far_masked_image = cv2.bitwise_and(image,mask) #with bitwise AND we choose only lines in Region of Interest
    return far_masked_image

def far_ROI_left(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([            # defining the dimensions for the region of interest
    [(350, 720),(470, 720),(470,370),(350, 370)] #[(300, 450),(900, 450),(700,350),(500, 350)]
    ])
    mask = np.zeros_like(image)  # make an black image with dimensions of the input image
    cv2.fillPoly(mask, polygons, 250) # apply white(250) on mask with dimensions defined on polygons
    far_masked_image = cv2.bitwise_and(image,mask) #with bitwise AND we choose only lines in Region of Interest
    return far_masked_image

def far_ROI_right(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([            # defining the dimensions for the region of interest
    [(470, 720),(470+120, 720),(470+120,370),(470, 370)] # [(300, 450),(900, 450),(700,350),(500, 350)]
    ])
    mask = np.zeros_like(image)  # make an black image with dimensions of the input image
    cv2.fillPoly(mask, polygons, 250) # apply white(250) on mask with dimensions defined on polygons
    far_masked_image = cv2.bitwise_and(image,mask) #with bitwise AND we choose only lines in Region of Interest
    return far_masked_image

### ........................................................................................................................... ###
### Finding Lines Algorithm (Here as a test OpenCV internal H-Transform is used)
def lane_finding_algorithm(image):
    model = cv2.HoughLinesP(image, 2, np.pi/180, 100, np.array([]), minLineLength=82, maxLineGap=5) #40, max 82
    return model

### ........................................................................................................................... ###
# This funcction draw the detected line in RIO found by Algorithm on a
# Black image with dimension of original image
# (The result of H-Transform is a line between (x1,y1) and (x2,y2), which will be used in this function)
def display_lines(image, lines):
    line_image = np.zeros_like(image)  # we produce a black image with our original image dimensions
    if lines is not None:
        for line in lines:
            # print(line) # lines we have are a 2-D array
            x1,y1,x2,y2 = line.reshape(4)  # we reshape the 2-D array to 1-D array
            cv2.line(line_image, (x1,y1), (x2,y2), (250, 0, 0), 10)     # this brings the line we found with H-Transforms (Mathematical Model)
                                                                        # on the black image we defined at the beggining of function, line thickness=10
    return line_image

### ........................................................................................................................... ###
### This function makes the Fitting,
### and gives an average line from lines detected with H-Transform and we have them in right and left side of the road:
### (lines is model from lane_finding_algorithm, this function finds the fitting parameters first,
### then with help of make_coordinates function finds the fitted model(line)):
def near_average_slope_intercept(image, lines):
    left_fit = [] # left line
    right_fit = [] # right line
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)  # Parameters of the found mathematical model
        parameters = np.polyfit((x1,x2), (y1,y2), 1) # return a vector of coefficient which describes the slope and intercept of the line for these 2 points
        slope = parameters[0]
        intercept = parameters[1]  # Till here the paramaters of the fit have been found!
        if slope < 0:              # find out the found fit parameteres for each line belong to left lane or right lane:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)  # gives averaged slope and intercept for left
    right_fit_average = np.average(right_fit, axis=0)  #  gives averaged slope and intercept for right
    near_left_line = near_make_coordinates(image, left_fit_average) # since we don't have the coordinates of beggining and end of line, we use make_coordinates function as below
    near_right_line = near_make_coordinates(image, right_fit_average)
    return np.array([near_left_line, near_right_line])

def far_average_slope_intercept(image, lines):
    left_fit = [] # left line
    right_fit = [] # right line
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)  # Parameters of the found mathematical model
        parameters = np.polyfit((x1,x2), (y1,y2), 1) # return a vector of coefficient which describes the slope and intercept of the line for these 2 points
        slope = parameters[0]
        intercept = parameters[1]  # Till here the paramaters of the fit have been found!
        if slope < 0:              # find out the found fit parameteres for each line belong to left lane or right lane:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)  # gives averaged slope and intercept for left
    right_fit_average = np.average(right_fit, axis=0)  #  gives averaged slope and intercept for right
    far_left_line = far_make_coordinates(image, left_fit_average) # since we don't have the coordinates of beggining and end of line, we use make_coordinates function as below
    far_right_line = far_make_coordinates(image, right_fit_average)
    return np.array([far_left_line, far_right_line])

### ........................................................................................................................... ###
### Near Field Finding the fitted line based on the fitting parametters got from the fitting step
def near_make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] # height of image
    y2 = int(y1*(3/5))  # estimate from the image we right_fit_average
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

### Far Field Finding the fitted line based on the fitting parametters got from the fitting step
def far_make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] - 350 # height of image
    y2 = int(y1*(3/5))  # estimate from the image we right_fit_average
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

### ........................................................................................................................... ###
### This function finds slope and intercepts of a line with two points of it given
### the line is as follows: y = mx + c
def slope_intercept_finder(line_set):
    x1 = line_set[0]
    y1 = line_set[1]
    x2 = line_set[2]
    y2 = line_set[3]

    slope = (y1-y2)/(x1-x2)
    intercept = ((y2*x1)-(y1*x2))/(x1-x2)
    return slope, intercept

### ........................................................................................................................... ###
### This Function gives a bird eye view of an image and the reverse view of lef and right side lanes:
def BirdEyeView(img):
    width , height = 600, 400
    pts1 = np.float32([[400, 390],[670,390],[350, 450],[720, 450]])  # top-left, top-right, bottom-left, bottom-right
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    Matrix = cv2.getPerspectiveTransform(pts1, pts2)
    Minv = cv2.getPerspectiveTransform(pts2, pts1)

    warped = cv2.warpPerspective(img, Matrix, (width,height))

    for x in range(0,4):
        circle = cv2.circle(img, (pts1[x][0],pts1[x][1]),5,(100,125,255), cv2.FILLED)

    copy_img_right = np.copy(warped)
    copy_img_left = np.copy(warped)
    H, L = np.shape(warped)
    # print('L = ', L, '\t' 'H = ', H)
    dividing_point = []
    for i in range(H):
        non_zero_values = np.nonzero(warped[i])
        sum_values = np.sum(np.nonzero(warped[i]))
        count_values = np.count_nonzero(warped[i])
        res_sum = sum_values/count_values
        res_sum = int(res_sum)
        dividing_point.append(res_sum)
        dividing_line = cv2.circle(warped, (dividing_point[i], i),3,(100,125,255), cv2.FILLED)

    cv2.line(warped, (dividing_point[0], 0), (dividing_point[399],399), (255,0,0) , thickness = 2)
    #------------------------------------------------------------------------------
    ### Cutting two sides of the line
    ### Left side of the road:
    point1 = (dividing_point[0],0)
    point2 = (dividing_point[399],H)
    polygons_left = np.array([
    [(0, 0),point1,point2,(0,H)]
    ])
    mask_left = np.zeros_like(copy_img_left)  # make an black image with dimensions of the input image
    cv2.fillPoly(mask_left, polygons_left, 250) # apply white(250) on mask with dimensions defined on polygons
    left_masked_image = cv2.bitwise_and(copy_img_left,mask_left) #with bitwise AND we choose only lines in Region of Interest

    ## Right side of the road:
    polygons_right = np.array([
    [point1,(L,0),(L,H),point2]
    ])
    mask_right = np.zeros_like(copy_img_right)  # make an black image with dimensions of the input image
    cv2.fillPoly(mask_right, polygons_right, 250) # apply white(250) on mask with dimensions defined on polygons
    right_masked_image = cv2.bitwise_and(copy_img_right,mask_right) #with bitwise AND we choose only lines in Region of Interest

    img_inv = cv2.warpPerspective(warped, Minv, (img.shape[1], img.shape[0]))  # width , height of the input image
    img_inv_left = cv2.warpPerspective(left_masked_image, Minv, (img.shape[1], img.shape[0]))  # width , height of the input image
    img_inv_right = cv2.warpPerspective(right_masked_image, Minv, (img.shape[1], img.shape[0]))  # width , height of the input image

    return warped, img_inv, dividing_point, right_masked_image, left_masked_image, img_inv_left, img_inv_right, circle

### ........................................................................................................................... ###
### ............................................. Processing the image:........................................................ ###
### ........................................................................................................................... ###

# ---------------------------------------------------------------------------------------------------------------------------------------#
print('\n','----------------------- Near Field Part ------------------------------------')
### 1.1. Loading Image and get the gradient image by canny:
original_image = cv2.imread('curve_left.jpg') # Load the image
# original_image = cv2.imread('webots_pictures/straight.jpg') # Load the image
image = np.copy(original_image) #Make a copy of of our original image
canny = canny(image)  # get the canny image of the image
near_ROI_image = near_ROI(canny) # get the near-field part of the canny image
# cv2.imshow('Test', canny)

### 1.2. Finding the points of the model by our Algorithm:
near_lines = lane_finding_algorithm(near_ROI_image)

# print('near_lines_cordiantes = ','\n', near_lines, '\n\n')
# print('far_lines_cordinates = ', far_lines)

### 1.3. Merging the detected model by algorithm on a black image and the original image
near_lines_image_Hough = display_lines(image, near_lines)  # bring the lines detected by algoritm on the original image
near_combo_image_Hough = cv2.addWeighted(image, 0.8, near_lines_image_Hough, 1, 1)


### 1.4. Optimizing the result:
# Finding the parameters for a fitted model which is ok for the found points of the model in step 2:
averaged_near_lines = near_average_slope_intercept(image, near_lines) # This function gives only one average line at right and one average line at left from detected lines by H-transform
first_line = averaged_near_lines[0]
second_line = averaged_near_lines[1]

first_line_params = slope_intercept_finder(averaged_near_lines[0])  # First fitted line (right-side lane)
slope1 = first_line_params [0]
intercept1 = first_line_params[1]
second_line_params = slope_intercept_finder(averaged_near_lines[1]) # 2nd fitted line (left-side lane)
slope2 = second_line_params[0]
intercept2 = second_line_params[1]

print('\n\n','Found fitted lines in near field = ', '\n',averaged_near_lines)
print('\n\n','first line =', '\n', first_line)
print('\n\n','second line =', '\n', second_line)
print('\n\n','first line params =', '\n', first_line_params)
print('\n\n','second line params =', '\n', second_line_params)
print('\n\n','1st line slope =', '\n', slope1)
print('\n\n','1st line intercept =', '\n', intercept1)
print('\n\n','2nd line slope =', '\n', slope2)
print('\n\n','2nd line intercept =', '\n', intercept2)


# Merging the fitted model on a black image and the original image:
near_lines_image_Av = display_lines(image, averaged_near_lines)  # bring the averaged lines on the original image
near_combo_image_Av = cv2.addWeighted(image, 1, near_lines_image_Av, 1, 1) # bring the averaged lines on a black image

# print('Average near_lines_cordiantes = ','\n', near_lines_image_Av, '\n\n')


# ---------------------------------------------------------------------------------------------------------------------------------------#
print('\n','----------------------- Far Field Part ------------------------------------')

BirdView_result = BirdEyeView(canny)
BirdView = BirdView_result[0]
imageInverseTotal = BirdView_result[1]
dividing_point = BirdView_result[2]
right_BirdEyeView =BirdView_result[3]
left_BirdEyeView = BirdView_result[4]
leftSide_inv = BirdView_result[5]
rightSide_inv = BirdView_result[6]
img_with_pts = BirdView_result[7]


### 2. Far field Operations on left side:

### 2.1. Get the gradient image by canny on left side:
far_ROI_image_left = leftSide_inv

#### 2.2. Getting the Data-set:
### 2.2.1. Finding the indeces of nonezero values on edge image:
index_left = np.nonzero(far_ROI_image_left) # Indices of Edge points on edge image
columns_left = index_left[1]
rows_left = index_left[0]
k_left = len(rows_left)

#### 2.2.2. Store the coordinates of the points of lanes found on the edge image:
dataSet_left = []
x_values_left = []
y_values_left = []
for ii in range(len(rows_left)):
    dataSet_left.append((rows_left[ii], columns_left[ii]))
    x_values_left.append(rows_left[ii])
    y_values_left.append(columns_left[ii])


### 2.3. Fitting the curved lanes with ransac:
x_values_arr_left = np.array(x_values_left)
X_left = x_values_arr_left[:, np.newaxis]
y_left = np.array(y_values_left)


# Robustly fit linear model with RANSAC algorithm

print('\n ------------------ Start of Ransac Algorithm Left: -------------- \n\n')
ransac_left = linear_model.RANSACRegressor()
a_left = ransac_left.fit(X_left, y_left)
inlier_mask_left = ransac_left.inlier_mask_
outlier_mask_left = np.logical_not(inlier_mask_left)

# Predict data of estimated models
line_X_left = np.arange(X_left.min(), X_left.max())[:, np.newaxis]
line_y_ransac_left = ransac_left.predict(line_X_left)

### 2.4. Fitting the curved lanes with polynomial:

### Linear Regression:
lin_reg_left = LinearRegression()
lin_reg_left.fit(X_left,y_left)

### Linear-Polynomial Regression:
poly_reg_left = PolynomialFeatures(degree=2)
X_poly_left = poly_reg_left.fit_transform(X_left)
lin_reg_2_left = linear_model.RANSACRegressor() # RANSAC Fitting of Polynomial Data
lin_reg_2_left.fit(X_poly_left, y_left)

### Plotting:
fig1_left =plt.figure(num='1. Left Far field image')
plt.title('Left Far field edge image')
plt.imshow(far_ROI_image_left)
fig1_left.show()
#------------------------------------------------------
fig2_left =plt.figure(num='2. Our Left Data Set')
for i in range(len(dataSet_left)):
    plt.plot(dataSet_left[i][1], dataSet_left[i][0], 'g*')
plt.xlim(0,1280)
plt.ylim(0,720)
plt.title('Data Set')
plt.gca().invert_yaxis()
fig2_left.show()
#------------------------------------------------------
fig3_left = plt.figure(num='3. Bring the Left data set into Log form')
#fig3_left = plt.figure(num='Left side data set')
plt.title('Left side data set')
for i in range(len(dataSet_left)):
    plt.plot(dataSet_left[i][1], np.log(dataSet_left[i][0]), 'r*')

plt.gca().invert_yaxis()
fig3_left.show()
#------------------------------------------------------
fig4_left = plt.figure(num = 'Left side RANSAC linear fit')
lw = 2
plt.scatter( y_left[inlier_mask_left],X_left[inlier_mask_left], color='yellowgreen', marker='.',
            label='Inliers Left')
plt.scatter( y_left[outlier_mask_left],X_left[outlier_mask_left], color='gold', marker='.',
            label='Outliers Left')
plt.plot( line_y_ransac_left,line_X_left, color='cornflowerblue', linewidth=lw,
         label='Linear RANSAC Left')

plt.gca().invert_yaxis()
plt.legend(loc='lower right')
plt.title('Left side RANSAC linear fit')


# #------------------------------------------------------
# fig5_left = plt.figure(num= '5.1. Linear-Polynomial Regression Left')
fig5_left = plt.figure(num= '5.Left side Polynomial-RANSAC fit')
plt.scatter(y_left, X_left, color = 'red', label='Left side Data set')
# plt.plot(X_left, line_reg.predict_left(X_left), color = 'blue')
plt.plot(lin_reg_2_left.predict(poly_reg_left.fit_transform(X_left)), X_left,  color = 'green', linewidth=5 , label='Polynomial-RANSAC fit')
plt.title('Linear and Linear-Polynomial Regression Left')
plt.gca().invert_yaxis()
plt.title('Left side Polynomial-RANSAC fit')
plt.legend(loc = 'lower right')

# ----------------------------------------------------------------------------------------------------- #

### 3. Far field Operations on right side:


### 3.1. Get the gradient image by canny on right side:
far_ROI_image_right = rightSide_inv

#### 3.2. Getting the Data-set:
### 3.2.1. Finding the indeces of nonezero values on edge image:
index_right = np.nonzero(far_ROI_image_right) # Indices of Edge points on edge image
columns_right = index_right[1]
rows_right = index_right[0]
k_right = len(rows_right)

#### 3.2.2. Store the coordinates of the points of lanes found on the edge image:
dataSet_right = []
x_values_right = []
y_values_right = []
for ii in range(len(rows_right)):
    dataSet_right.append((rows_right[ii], columns_right[ii]))
    x_values_right.append(rows_right[ii])
    y_values_right.append(columns_right[ii])



### 3.3. Fitting the curved lanes with ransac:
x_values_arr_right = np.array(x_values_right)
X_right = x_values_arr_right[:, np.newaxis]
y_right = np.array(y_values_right)

# Robustly fit linear model with RANSAC algorithm

print('\n ------------------ Start of Ransac Algorithm right: -------------- \n\n')
ransac_right = linear_model.RANSACRegressor()
a_right = ransac_right.fit(X_right, y_right)
inlier_mask_right = ransac_right.inlier_mask_
outlier_mask_right = np.logical_not(inlier_mask_right)

# Predict data of estimated models
line_X_right = np.arange(X_right.min(), X_right.max())[:, np.newaxis]
line_y_ransac_right = ransac_right.predict(line_X_right)

### 3.4. Fitting the curved lanes with polynomial:

### Linear Regression:
lin_reg_right = LinearRegression()
lin_reg_right.fit(X_right,y_right)

### Linear-Polynomial Regression:
poly_reg_right = PolynomialFeatures(degree=2)
X_poly_right = poly_reg_right.fit_transform(X_right)
poly_reg_right.fit(X_poly_right, y_right)
lin_reg_2_right = linear_model.RANSACRegressor() # RANSAC Fitting of Polynomial Data
lin_reg_2_right.fit(X_poly_right, y_right)

### Plotting:
fig1_right =plt.figure(num='1. right Far field image')
plt.title('right Far field edge image')
# plt.imshow(far)
plt.imshow(far_ROI_image_right)
fig1_right.show()
#------------------------------------------------------
fig2_right =plt.figure(num='2. Our right Data Set')
for i in range(len(dataSet_right)):
    plt.plot(dataSet_right[i][1], dataSet_right[i][0], 'g*')
plt.xlim(0,1280)
plt.ylim(0,720)
plt.title('Data Set')
plt.gca().invert_yaxis()
fig2_right.show()
#------------------------------------------------------
fig3_right = plt.figure(num='3. Bring the right data set into Log form')
# fig3_right = plt.figure(num='Right side data set')
plt.title('Right side data set')
for i in range(len(dataSet_right)):
    plt.plot(dataSet_right[i][1], np.log(dataSet_right[i][0]), 'r*')
plt.gca().invert_yaxis()
fig3_right.show()
#------------------------------------------------------
fig4_right = plt.figure(num = 'Right side RANSAC linear fit')
lw = 2
plt.scatter( y_right[inlier_mask_right],X_right[inlier_mask_right], color='yellowgreen', marker='.',
            label='Inliers right')
plt.scatter( y_right[outlier_mask_right],X_right[outlier_mask_right], color='gold', marker='.',
            label='Outliers right')
plt.plot( line_y_ransac_right,line_X_right, color='cornflowerblue', linewidth=lw,
         label='Linear RANSAC Right')

plt.gca().invert_yaxis()
plt.legend(loc='lower right')
plt.title('Right side RANSAC linear fit')

# #------------------------------------------------------
fig5_right = plt.figure(num= '5.Right side Polynomial-RANSAC fit')
plt.scatter(y_right, X_right, color = 'red', label='right Data set')
plt.plot(lin_reg_2_right.predict(poly_reg_right.fit_transform(X_right)), X_right,  color = 'green', linewidth=5 , label='Polynomial-RANSAC fit')
plt.title('Right side Polynomial-RANSAC fit')
plt.gca().invert_yaxis()
plt.title('Right side Polynomial-RANSAC fit')
plt.legend(loc = 'lower right')

### ........................................................................................................................... ###

# ### 4. Showing the Results:

### 1. Showing the Results in near field:
# cv2.imshow('Original Image',image)  # Shows the main image
# cv2.imshow('Canny Image', canny)  # Shows the canny image
# cv2.imshow('Near Field ROI',near_ROI_image)  # Shows near field of canny image
# plt.imshow(near_ROI_image)

# ### Showing Detected Lines by Algorithm on a black image and the original image
# cv2.imshow('Detected H-Transform Lines on near field - Black Image',near_lines_image_Hough) # Shows the H-Transform detected lines on a black image
# cv2.imshow('Detected H-Transform Lines on near field - Combo',near_combo_image_Hough) # Shows the H-Transform detected lines on the original Image

### Showing Detected optimized Lines by Algorithm on a black image and the original image
# cv2.imshow('Detected Averaged Lines (Fit Model) on near field - Black Image',near_lines_image_Av) # Shows the H-Transform detected lines on a black image
# cv2.imshow('Detected Averaged Lines (Fit Model) on near field - Combo',near_combo_image_Av) # Shows the H-Transform detected lines on the original Image

# plt.imshow(near_combo_image_Av)


### 2. Showing results in far field:
#### cv2.imshow('Far Field ROI',far_ROI_image_left)  # Shows far field of canny image
# # cv2.imshow('Detected Averaged Lines (Fit Model) on far field - Black Image',far_lines_image_Av) # Shows the H-Transform detected lines on a black image
# # cv2.imshow('Detected Averaged Lines (Fit Model) on far field - Combo',far_combo_image_Av) # Shows the H-Transform detected lines on the original Image
# fig6 = plt.figure(num= '6. Far Field Fit')
# plt.plot(lin_reg_2_right.predict(poly_reg_right.fit_transform(X_right)), X_right,  color = 'yellow', linewidth=5 , label='Polynomial Regression right')
# plt.plot(lin_reg_2_left.predict(poly_reg_left.fit_transform(X_left)), X_left,  color = 'green', linewidth=5 , label='Polynomial Regression Left')
# plt.imshow(image)
# # plt.imshow(img_with_pts)
# plt.imshow(near_combo_image_Av)
# # plt.gca().invert_yaxis()
# plt.title('Linear-Polynomial Regression right')
# plt.legend(loc = 'lower right')
# plt.xlabel("Input right")
# plt.ylabel("Response right")

param_right = lin_reg_2_right.predict(poly_reg_right.fit_transform(X_right))
param_right = param_right.astype(np.int)
for iii in range(len(X_right)-1):
    cv2.line(image, (param_right[iii],X_right[iii]), (param_right[iii+1],X_right[iii+1]), (255,0,0), thickness=10)

param_left = lin_reg_2_left.predict(poly_reg_left.fit_transform(X_left))
param_left = param_left.astype(np.int)
for iii in range(len(X_left)-1):
    cv2.line(image, (param_left[iii],X_left[iii]), (param_left[iii+1],X_left[iii+1]), (255,0,0), thickness=10)

# X_mid = (X_left[iii] + X_right[iii])/2
# param_mid = (param_right+param_left)/2
# param_mid = param_mid.astype(np.int)
# for iii in range(len(X_left)-1):
#     cv2.line(image, (param_left[iii],X_mid[iii]), (param_left[iii+1],X_mid[iii+1]), (255,0,0), thickness=10)


########################################################################################################

# ### 3. Showing Total result:
final_result_img = cv2.addWeighted(image, 1, near_lines_image_Av, 1, 1) # bring the averaged lines on a black image
cv2.imshow('Final Result', final_result_img)
print('Everything worked ok. \n\n\n')
#print(original_image.shape)

cv2.waitKey(0)
input()
plt.show()
