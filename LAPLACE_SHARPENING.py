import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

img=plt.imread(" ")
plt.imshow(img)

grayimg=np.dot(img[:,:,:3],[0.2989,0.5870,0.1140])
plt.imshow(grayimg,cmap='gray')

grayimgarr=np.array(grayimg)
grayimgarr.shape

grayimgarr=grayimgarr.reshape(2160, 3840,1)
grayimgarr.shape

def convolution(image,kernel):
#     mode=valid
    rows,cols,channels=image.shape
    m,n= kernel.shape
    ystride=rows-m+1
    xstride=cols-n+1
    
    new_img=np.zeros((rows,cols,channels))
#     new_img_temp=new_img.flatten()
#     new_img_temp=new_img_temp.reshape(-1,1)
    print(new_img.shape)
    for i in range(ystride):
        for j in range(xstride):
            for c in range(channels):
#                 c will be 1 only as gray image
                sub_matrix=image[i:i+m,j:j+n,c]
                new_img[i,j,c] = np.sum(sub_matrix * kernel)
    return new_img

def laplacian_filter(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return convolution(image, kernel)

laplacian_img=laplacian_filter(grayimgarr)

ShF         = 100                   #Sharpening factor!
Laps        = laplacian_img*ShF/np.amax(laplacian_img) 

sharpened_image = grayimgarr + Laps

plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(sharpened_image,cmap='gray')
plt.title('Laplacian Sharpened  Image')

sharpened_image = grayimgarr - Laps

sharpened_image.min()

plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.imshow(grayimg,cmap='gray')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(sharpened_image,cmap='gray')
plt.title('Laplacian Sharpened  Image')
