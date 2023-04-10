import cv2
import numpy as np
import cv2 as cv


def main():
    H1 = np.load("../rectification/H1.npy")
    H2 = np.load("../rectification/H2.npy")

    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    if not cap.isOpened() or not cap2.isOpened():
        print("Cannot open camera")
        exit()
    count = 0

    def nothing(x):
        pass

    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Disparity', 450, 450)
    cv2.createTrackbar('max_disp', 'Disparity', 1, 16, nothing)
    cv2.createTrackbar('min_disp', 'Disparity', 0, 32, nothing)
    cv2.createTrackbar('blockSize', 'Disparity', 5, 50, nothing)
    cv2.createTrackbar('uniquenessRatio', 'Disparity', 15, 100, nothing)
    cv2.createTrackbar('speckleRange', 'Disparity', 0, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', 'Disparity', 3, 200, nothing)
    cv2.createTrackbar('disp12MaxDiff', 'Disparity', 0, 25, nothing)

    while True:
        # Capture frame-by-frame
        ret, img1 = cap.read()
        ret2, img2 = cap2.read()
        h1, w1 = img1.shape[:2]
        h2, w2 = img1.shape[:2]

        # Updating the parameters based on the trackbar positions
        max_disp = cv2.getTrackbarPos('max_disp', 'Disparity') * 16
        min_disp = (cv2.getTrackbarPos('min_disp', 'Disparity') - 16) * 16
        block_size = cv2.getTrackbarPos('blockSize', 'Disparity') * 2 + 5
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'Disparity')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'Disparity')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'Disparity') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'Disparity')

        img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
        img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))

        num_disp = max_disp - min_disp

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )

        disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)

        imgg = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        imgg = cv2.resize(imgg, (320, 240))
        disparity_SGBM = cv2.resize(disparity_SGBM, (320, 240))

        # Display the resulting frame
        img_concate_Hori = np.concatenate((imgg, disparity_SGBM), axis=0)
        cv2.imshow('Output', img_concate_Hori)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('s'):
            print("Saving image")
            cv2.imwrite("./testData/" + str(count) + "L.jpg", img1)
            cv2.imwrite("./testData/" + str(count) + "R.jpg", img2)
            count = count + 1
    # When everything done, release the capture
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
