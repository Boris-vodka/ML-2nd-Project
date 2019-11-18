from PIL import Image
from PIL import ImageFilter
import math
import cv2
from astropy.visualization import LogStretch, MinMaxInterval,ImageNormalize
import matplotlib.pyplot as plt
import numpy as np
import glob
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import LogStretch, MinMaxInterval,ImageNormalize

# ==============================================================================
# Data Loading
# ==============================================================================
def load_images (path, extensions , load_length , IR = False , test = False, start = 0):

    print('Data loading :')

    if test == False :
        start = 0;

    if IR:

        transform =  MinMaxInterval()

        filelist_y = sorted(glob.glob( path + 'EUC_Y/' + extensions))
        filelist_j = sorted(glob.glob( path + 'EUC_J/' + extensions))
        filelist_h = sorted(glob.glob( path + 'EUC_H/' + extensions))

        Images = np.empty( ( load_length,200,200 ) ,dtype=np.float32)

        for k in range(start,load_length):
            if k%300==0:
                print(k)

            Im_y = np.array(fits.getdata( filelist_y[k+start]))
            Im_y = transform(Im_y)
            Im_y= np.array( cv2.resize(Im_y, dsize=(200, 200))  )

            Im_j = np.array(fits.getdata( filelist_j[k+start]))
            Im_j = transform(Im_j)
            Im_j= np.array( cv2.resize(Im_j, dsize=(200, 200))  )

            Im_h = np.array(fits.getdata( filelist_h[k+start]))
            Im_h = transform(Im_h)
            Im_h= np.array( cv2.resize(Im_h, dsize=(200, 200))  )

            Images[k,:,:] = 0.15 * Im_y + 0.35*Im_j + 0.5 * Im_h

    else:

        filelist = sorted(glob.glob( path + 'EUC_VIS/' + extensions))

        Images = np.empty( ( load_length,200,200 ) ,dtype=np.float32)

        for k in range(start,load_length):
            if k%1000==0:
                print(k)
            Images[k,:,:] = np.array( fits.getdata( filelist[k+start] ) )

    print('Done loading data')

    return Images


def load_csv_data(data_path, col ):
    """Loads data for a certain column"""

    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    input_data = x[1:, col]

    return input_data

# ==============================================================================
# Image combination
# ==============================================================================
def Img_combine (Vis_set , IR_set):

    Images = np.empty( (Vis_set.shape[0],200,200,2 ) ,dtype=np.float32)

    for k in range(Vis_set.shape[0]):
        Images[k,:,:,0] = Vis_set[k,:,:]
        Images[k,:,:,1] = IR_set[k,:,:]

    return Images

# ==============================================================================
# Data trasnformation
# ==============================================================================

def add_dim (image_set1, image_set2):

    return np.expand_dims(image_set1, axis=3), np.expand_dims(image_set2, axis=3)

# ==============================================================================
# Image trasnformation
# ==============================================================================

def logarithmic_scale (Images, Images_load_length):

    transform = LogStretch() + MinMaxInterval()  # def of transformation

    print('Logarithmic strech :')

    for k in range(Images_load_length):
        if k%1000==0:
            print(k)
        Images[k,:,:] = transform(Images[k,:,:])

    print('Logarithmic data strech done')

    return Images

# Compute log transform of the image
def logTransform(c, f):
    g = c * math.log(float(1 + f),10)
    return g

# Apply logarithmic transformation for an image
def logTransformImage(image, outputMax = 255, inputMax=255):

    c = outputMax/math.log(inputMax+1,10);

    # Read pixels and apply logarithmic transformation
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            # Get pixel value at (x,y) position of the image
            f = image[i,j]

            # Do log transformation of the pixel
            logPixel  = round(logTransform(c, f));

            # Modify the image with the transformed pixel values
            image[i,j] = logPixel

    return image

# ==============================================================================
# Image plotting
# ==============================================================================

def plot_images_test (train_images,norm,N_im) :


    plt.imshow(train_images[N_im],origin='lower', norm=norm)
    plt.show()


def plot_images_test_mosaic (train_images,norm) :

    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i],origin='lower', norm=norm)
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        # plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
