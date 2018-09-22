import cv2
import numpy as np
import sys


def rotate(img, angle, landmarks):
    """Rotate the passed image by the angle specified counterclockwise. Also calculates the new coordinates in the image for the landmarks

    Args:
        img (OpenCV Mat): valid image with one or three channels
        angle (float): degrees to rotate the image counterclockwise
        landmarks: (numpy array): array of landmarks on the image

    Returns:
        tuple: contains the rotated image, as well as the coordinates of the landmarks after rotation 
    """
    width, height = img.shape[:2]
    rotation = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation, (width, height))

    rotated_landmarks = np.asarray([np.dot(rotation, landmark.T) for landmark in landmarks])
    return rotated_img, rotated_landmarks


if __name__ == '__main__':
    """
    sys.argv[0] = this script
    sys.argv[1] = location to image to rotate
    sys.argv[2] = angle (in degrees) to rotate the image
    """
    if (len(sys.argv) != 3):
        raise ValueError('must provide 3 command line arguments')

    img = cv2.imread(sys.argv[1]) 
    landmarks = np.asarray([[100, int(img.shape[0]/2), 1], [int(img.shape[1]/2), int(img.shape[0]/2), 1]]) #sample spots for landmarks

    for points in landmarks:
        cv2.circle(img, (points[0], points[1]), 6, (0, 0, 255), -1)
    
    cv2.imshow('original', img) #show original image
    img, rotated_landmarks = rotate(img, sys.argv[2], landmarks) 

    cv2.imshow('image', img) #show rotated image
    cv2.waitKey()
    cv2.destroyAllWindows()
