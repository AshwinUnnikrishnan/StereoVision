import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)

    sift = cv2.xfeatures2d.SIFT_create()

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if not cap.isOpened() or not cap2.isOpened():
        print("Cannot open camera")
        exit()
    count = 0
    while True:
        # Capture frame-by-frame
        ret, img1 = cap.read()
        ret2, img2 = cap2.read()

        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.70 * n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(keypoints_2[m.trainIdx].pt)
                pts1.append(keypoints_1[m.queryIdx].pt)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask[300:500],
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        keypoint_matches = cv2.drawMatchesKnn(
            img1, keypoints_1, img2, keypoints_2, matches[300:500], None, **draw_params)
        keypoint_matches = cv2.resize(keypoint_matches, (1000, 400))

        if len(pts1) >= 8:
            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
            # We select only inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            print("saving image", len(pts1), count)
            cv2.imwrite("./data/best_pairs/" + str(count) + "L.jpg", img1)
            cv2.imwrite("./data/best_pairs/" + str(count) + "R.jpg", img2)
            count = count + 1

        # # Display the resulting frame
        cv2.imshow('best_pairs', keypoint_matches)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
