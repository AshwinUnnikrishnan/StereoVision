# Created calibration.py by histravelstories on 6:47 PM under Project6

import cv2 as cv
import numpy as np
import config
from utils import checkChessBoard, calibrateAndStore
from loggingCalib import get_logger

logger = get_logger(__name__)


def main():
    """Starting the video capture sequence
    camR is the webcam"""

    camR = cv.VideoCapture(config.CAMERA1)
    camL = cv.VideoCapture(config.CAMERA2)
    calibrationImageCount = 1  # To store the number of images used for calibration

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objPoints = []  # 3d point in real world space
    imgPointsL = []  # 2d points in image plane.
    imgPointsR = []  # 2d points in image plane.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    logger.debug("Starting operations")
    while True:
        retR, frameR = camR.read()
        retL, frameL = camL.read()

        grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
        grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)

        operation = cv.waitKey(1)
        if operation == ord('q'):  # quits the infinite loop
            logger.info("Quitting the loop")
            break
        elif operation == ord('s'):
            logger.info("Checking if corners are found")
            status, retL, cornerL, retR, cornerR = checkChessBoard(grayL, grayR, calibrationImageCount)
            if status:
                calibrationImageCount += 1
                imgPointsL.append(cornerL)
                imgPointsR.append(cornerR)
                objPoints.append(objp)
                cv.drawChessboardCorners(frameL, (9, 6), cornerL, retL)
                cv.drawChessboardCorners(frameR, (9, 6), cornerR, retR)

        elif operation == ord('c'):
            if calibrationImageCount >= 8:
                logger.info("Starting Camera Calibration")
                calibrateAndStore(objPoints, grayL, grayR, imgPointsL, imgPointsR)

        cv.imshow('VideoR', frameR)
        cv.imshow('VideoL', frameL)

    camR.release()
    camL.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
