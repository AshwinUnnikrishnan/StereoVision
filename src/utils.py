# Created utils.py by histravelstories on 7:16 PM under Project6


import os
import cv2 as cv
import pickle
from loggingCalib import get_logger
import config
logger = get_logger(__name__)


def storeImages(location, grayL=None, grayR=None, calibrationImageCount=None):
    """
    creates a folder and stores the left image and the right image used for camera calibration
    :param location: location to store the images
    :param grayL: Grayscale version of left image
    :param grayR: Grayscale version of right image
    :param calibrationImageCount: Count of images
    :return:
    """
    if not os.path.exists(location):
        logger.debug("Folder {0} not present  not present, creating the folder".format(location))
        os.makedirs(location)
    logger.info("Storing the images {0}".format(calibrationImageCount))
    cv.imwrite(location + '/LeftImage_' + str(calibrationImageCount) + '.jpg', grayL)
    cv.imwrite(location + '/RightImage_' + str(calibrationImageCount) + '.jpg', grayR)

    return True


def checkChessBoard(grayL, grayR, calibrationImageCount):
    """
    to check if chessboard was found, if found by both left and right camera then it stores the image for calibration and returns the corner points
    :param grayL: left image
    :param grayR: right image
    :return: status : boolean variable to specify if both camera detected, along with the corner points for both the cameras
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    logger.info("Checking for chessboard corners")
    retL, cornersL = cv.findChessboardCorners(grayL, (9, 6),
                                              cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)
    retR, cornersR = cv.findChessboardCorners(grayR, (9, 6),
                                              cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)
    status = False
    if retL == True and retR == True:
        corners2L = cv.cornerSubPix(grayL, cornersL, (9, 6), (-1, -1), criteria)
        corners2R = cv.cornerSubPix(grayR, cornersR, (9, 6), (-1, -1), criteria)
        storeImages("../"+config.DATALOC, grayL, grayR, calibrationImageCount)

        return True, retL, corners2L, retR, corners2R
    if retL == False:
        logger.info("Left Camera did not detect chessboard")

    if retR == False:
        logger.info("Right Camera did not detect chessboard")
    return False, None, None, None, None


def calibrateAndStore(objPoints, grayL, grayR, imgPointsL, imgPointsR):
    """

    :param objPoints: 3D points
    :param grayL: grayscale image of left camera
    :param grayR: grayscale image of right camera
    :param imgPointsL: left camera corner points for chess
    :param imgPointsR: right camera corner points for chess
    :return:
    """
    retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objPoints, imgPointsL, grayL.shape[::-1], None, None)
    retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, grayR.shape[::-1], None, None)
    logger.info("Calibration Done, Storing the parameters in the pkl file")
    if not os.path.exists("../" + config.CALIBLOC):
        logger.debug("Folder calib not present  not present, creating the folder")
        os.makedirs("../" + config.CALIBLOC)

    with open("../" + config.CML, 'wb') as f:
        pickle.dump(mtxL, f)
    with open("../" + config.CMR, 'wb') as f:
        pickle.dump(mtxR, f)
    with open("../" + config.DCL, 'wb') as f:
        pickle.dump(distL, f)
    with open("../" + config.DCR, 'wb') as f:
        pickle.dump(distR, f)
    logger.debug(mtxL)
    logger.debug(mtxR)
    logger.debug(distL)
    logger.debug(distR)
