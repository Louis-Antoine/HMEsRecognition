"""
Create additional, distorted data from base data set
"""

import numpy as np
import glob
import cv2
import os

def distortData(input_data_path, output_data_path, iterations = 5, shrink_c = 0.15, shear_c = 0.1, rotation_c = 10, perspective_c = 0.05):
    file_list = glob.glob(input_data_path + "*")

    # create dir if it doesnt exist
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)

    # for each class in data
    for class_path in file_list:
        class_name = class_path.split("\\")[-1]

        # create class dir if it doesnt exist
        if not os.path.exists(output_data_path + class_name):
            os.makedirs(output_data_path + class_name)
        
        # for each image in class
        for index, img_path in enumerate(glob.glob(class_path + "\*.jpg"), 1):
            origin_image = cv2.imread(img_path, cv2.COLOR_RGB2GRAY)

            # write original image
            cv2.imwrite(output_data_path + class_name + "/" + class_name + "_" +str(index) + "_0.jpg", origin_image)

            for i in range(iterations):
                # randomly decide transforms with 4/7 chance to not transform twice
                first_transform_id = np.random.randint(0,3)
                second_transform_id = first_transform_id
                while second_transform_id == first_transform_id:
                    second_transform_id = np.random.randint(0,8)
                
                #initialize
                distorted_image = origin_image
            
                for id in (first_transform_id, second_transform_id):
                    if id == 0: # shrink
                        distorted_image = applyShrink(distorted_image, np.random.uniform(1 - shrink_c, 1 + shrink_c))

                    elif id == 1: # shear
                        distorted_image = applyShear(distorted_image, np.random.uniform(1 - shear_c, 1 + shear_c), 
                                                                      np.random.uniform(1 - shear_c, 1 + shear_c))

                    elif id == 2: # rotate
                        distorted_image = applyRotation(distorted_image, np.random.uniform(- rotation_c, rotation_c))

                    elif id == 3: # perspective
                        distorted_image = applyPerspective(distorted_image, np.random.uniform(1 - perspective_c, 1 + perspective_c), 
                                                                            np.random.uniform(1 - perspective_c, 1 + perspective_c))

                # cv2.imshow(class_name, distorted_image)
                # cv2.waitKey()
                cv2.imwrite(output_data_path + class_name + "/" + class_name + "_" + str(index) + "_" + str(i+1) + ".jpg", distorted_image)
        print("distorted " + str(index) + " " + class_name + " images")

def applyShrink(image, shrink_factor):
    width, height = image.shape[1], image.shape[0]
    shrink_m = np.float32([[shrink_factor, 0, width * (1-shrink_factor)/2],
                           [0, shrink_factor, height * (1-shrink_factor)/2],
                           [0, 0, 1]])

    distorted_image = cv2.warpPerspective(image, shrink_m, (width, height), borderValue = (255,255,255))
    return distorted_image

def applyShear(image, x, y):
    width, height = image.shape[1], image.shape[0]
    shear_in = np.float32([[0, 0],
                           [0, height],
                           [width, 0],
                           [width, height]])

    shear_out = np.float32([[0, 0],
                            [width * -(x - 1), height * y],
                            [width * x, height * -(y - 1)],
                            [width, height]])

    shear_matrix = cv2.getPerspectiveTransform(shear_in, shear_out)
    distorted_image = cv2.warpPerspective(image, shear_matrix,(width, height), borderValue = (255,255,255))
    return distorted_image

def applyRotation(image, theta):
    width, height = image.shape[1], image.shape[0]

    rotation_m = cv2.getRotationMatrix2D((width/2, height/2), theta, 1)
    distorted_image = cv2.warpAffine(image, rotation_m, (width, height), borderValue = (255,255,255))

    return distorted_image

def applyPerspective(image, a, b):
    width, height = image.shape[1], image.shape[0]
    perspective_in = np.float32([[0     , 0],
                                 [width , 0],
                                 [0     , height],
                                 [width , height]])

    perspective_out = np.float32([[width * (1 - a)  , height * (1 - b)],
                                  [width * a        , height * -(1 - b)],
                                  [width * -(1 - a) , height * b],
                                  [width * (2 - a)  , height * (2 - b)]])

    shear_matrix = cv2.getPerspectiveTransform(perspective_in, perspective_out)
    distorted_image = cv2.warpPerspective(image, shear_matrix,(width, height), borderValue = (255,255,255))
    return distorted_image