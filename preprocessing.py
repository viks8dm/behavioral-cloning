# preprocessing function for behavioral cloning project

# import necessary modules
import csv
import cv2
import scipy.misc
import numpy as np
import skimage.transform as sktransform
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# define some parameters to be used later
STEERING_CORRECTION = 0.25      # this is a parameter to tune

###################################
def read_single_image(img_name, file_path):
    # extract file path for local machine
    img_file = img_name.split('/')[-1]
    img_fullpath = file_path + img_file

    return img_fullpath

###################################
# read driving log data using all camera images
def read_and_add_training_data():
    csv_file = './data/driving_log.csv'
    car_images = []
    steering_angles = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # create adjusted steering str_angles for the side camera images
            steering_center = float(row[3])
            steering_left = steering_center + STEERING_CORRECTION
            steering_right = steering_center - STEERING_CORRECTION

            # read in images from center, left and right cameras
            center_img_filepath = read_single_image(row[0], './data/IMG/')
            left_img_filepath = read_single_image(row[1], './data/IMG/')
            right_img_filepath = read_single_image(row[2], './data/IMG/')

            img_center = cv2.imread(center_img_filepath)
            img_left = cv2.imread(left_img_filepath)
            img_right = cv2.imread(right_img_filepath)

            # add images and str_angles to data set
            car_images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])

            car_images.append(img_center)
            steering_angles.append(steering_center)

    # return np.array(car_images), np.array(steering_angles)


    # add mirror image to data set
    flip_images, flip_angles = flip_horz(car_images, steering_angles)

    # add sheared image to data set
    shear_images = []
    shear_angles = []
    for i in range (0, len(car_images)):
        image = car_images[i]
        str_ang = steering_angles[i]
        shear_image, shear_angle = random_shear(image, str_ang)
        shear_images.append(shear_image)
        shear_angles.append(shear_angle)

    car_images.extend(flip_images)
    steering_angles.extend(flip_angles)

    car_images.extend(shear_images)
    steering_angles.extend(shear_angles)

    car_images = crop_all_images(car_images)

    # resize image per NVIDIA paper
    all_images = []
    for image in car_images:
        all_images.append(resize_image(image))

    return np.array(all_images), np.array(steering_angles)

###################################
# flip images horizontally
def flip_horz(images, str_angles):
    augmented_images, augmented_str_angles = [], []
    for image,str_angle in zip(images, str_angles):
        # # reduce zero bias
        if np.absolute(str_angle)>0.1:
            # augmented_images.append(image)
            # augmented_str_angles.append(str_angle)
            augmented_images.append(cv2.flip(image,1))
            augmented_str_angles.append(-1.0 * str_angle)

    return np.array(augmented_images), np.array(augmented_str_angles)

##################################################
def resize_image(image):
    return scipy.misc.imresize(image, (66, 200, 3))

##################################################
# crop images
def crop_image(image):
    return image[70:135, :, :]

##################################################
# crop images
def crop_all_images(images):
    new_images = []
    for image in images:
        image = crop_image(image)
        new_images.append(image)
    return new_images

##################################################
# generate sheered image
def random_shear(image, steering_angle, shear_range=200):
    """
    Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle

##################################################
# apply gamma correction
def random_gamma(image):
    """
    Random gamma correction is used as an alternative method changing the brightness of
    training images.
    http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    new_image = cv2.LUT(image, table)

    return new_image

###################################
# function for processing training data images
def process_image(image):

    # image, steering_angle = random_shear(image, steering_angle)

    # image, steering_angle = random_flip(image, steering_angle)

    # add random gamma
    # image = random_gamma(image)

    # # crop image from top and bottom
    # image = image[50:140, :, :]

    # # resize image per NVIDIA paper
    # image = scipy.misc.imresize(image, (66, 200, 3))
    # new_image = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)

    # convert to YUV color
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

##################################
def process_data_set(images, angles):

    images, angles = shuffle(images, angles)

    mod_images = []
    mod_angles = []

    for image in images:
        proc_image = process_image(image)
        mod_images.append(proc_image)

    mod_angles = angles

    return np.array(mod_images), np.array(mod_angles)


###################################
############## main script
if __name__ == "__main__":
    images, str_angles = read_and_add_training_data()

    images, str_angles = process_data_set(images, str_angles)

    print("number of samples: ", len(images))
    print("number of str-samples: ", len(str_angles))
    print("image shape: ", images[0].shape)

    plt.hist(np.array(str_angles), bins='auto')
    plt.title("Steering angle distribution")
    plt.ylabel('number of images')
    plt.xlabel("steering angle")
    plt.show()



