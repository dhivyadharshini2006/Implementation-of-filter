# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1
Import the necessary libraries (cv2, numpy, matplotlib.pyplot) and read the input image using cv2.imread().
</br>
</br> 

### Step 2
Define the kernels for the filters, such as an averaging kernel for smoothing (e.g., a 5x5 matrix of ones divided by 25) and a Laplacian kernel for sharpening.
</br>
</br> 

### Step 3
Apply the smoothing filters to the original image using functions like cv2.filter2D() for the averaging filter, cv2.GaussianBlur() for Gaussian blur, and cv2.medianBlur() for median blur.
</br>
</br> 

### Step 4
Apply the sharpening filter using cv2.filter2D() with the Laplacian kernel, and then add this filtered output to the original image to create the final sharpened image.
</br>
</br> 

### Step 5
Display the original, smoothed, and sharpened images side-by-side using matplotlib.pyplot.imshow() and plt.subplot() for comparison.
</br>
</br> 

## Program
### Developed By   :Dhivya Dharshini B
### Register Number:212223240031
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```Python

import cv2
import matplotlib.pyplot as plt
import numpy as np
image1=cv2.imread("flower.jpg")
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
kernel=np.ones((11,11),np.float32)/169
image3=cv2.filter2D(image2,-1,kernel)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Average Filter Image")
plt.axis("off")
plt.show()





```
ii) Using Weighted Averaging Filter
```Python


kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image2=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
image3=cv2.filter2D(image2,-1,kernel1)
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()



```
iii) Using Gaussian Filter
```Python



gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()

```
iv)Using Median Filter
```Python


median=cv2.medianBlur(image2,13)
plt.figure(figsize=(9,9))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Median Blur")
plt.axis("off")
plt.show()


```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python



kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()

```
ii) Using Laplacian Operator
```Python


laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()


```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter

</br>
</br>
<img width="717" height="272" alt="download" src="https://github.com/user-attachments/assets/91728bf8-3479-41e1-9ef5-1e6e926d267a" />

ii)Using Weighted Averaging Filter

</br>
</br>
<img width="531" height="205" alt="download" src="https://github.com/user-attachments/assets/53b91ed4-46f8-481d-94c3-136ecff408d0" />

iii)Using Gaussian Filter

</br>
</br>
<img width="515" height="205" alt="download" src="https://github.com/user-attachments/assets/62b39284-c413-437f-9ef2-dc3163ff8f93" />

iv) Using Median Filter

</br>
</br>
<img width="717" height="272" alt="download" src="https://github.com/user-attachments/assets/859f6c1b-dde0-4ff1-b21b-50b016ee6fff" />

### 2. Sharpening Filters
</br>

i) Using Laplacian Kernal

</br>
</br>
<img width="515" height="205" alt="download" src="https://github.com/user-attachments/assets/36414416-fe2a-4f8c-aa01-a55e0cc097e4" />
ii) Using Laplacian Operator
</br>

</br>
</br>
<img width="515" height="205" alt="download" src="https://github.com/user-attachments/assets/a22c957b-383f-4600-96a2-6c11f82276d3" />

</br>
</br>

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
