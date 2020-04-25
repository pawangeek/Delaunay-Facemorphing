import cv2
import numpy as np


def delaunay(img, landmarks):

    # Doc : https://docs.opencv.org/master/df/dbf/classcv_1_1Subdiv2D.html
    # The function creates an empty subdivision where 2D points can be added
    subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))

    # Docs : https://docs.opencv.org/master/df/dbf/classcv_1_1Subdiv2D.html#a18a6c9999210d769538297d843c613f2
    # "inserts" a single point into a subdivision and modifies the subdivision topology appropriately.
    # If a point with the same coordinates exists already, no new point is added.
    for item in landmarks:
        subdiv.insert(item)

    # https://docs.opencv.org/master/df/dbf/classcv_1_1Subdiv2D.html#a26bfe32209bc8ae9ecc53e93da01e466
    # getTriangleList gives each triangle as a 6 numbers vector, where each two are one of the triangle vertices
    # p1x = v[0], p1y = v[1], p2x = v[2], p2y = v[3], p3x = v[4], p3y = v[5].
    return subdiv.getTriangleList()

def delaunay2(img, triangleIndex, landmarks, delaunay_color):

    for j in range(len(triangleIndex)):

        x, y, z = triangleIndex[j][0], triangleIndex[j][1], triangleIndex[j][2]
        pt1, pt2, pt3 = landmarks[x], landmarks[y], landmarks[z]

        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    return img


def getAffine(src, coord1, coord2, size):

    coord1, coord2 = np.array(coord1), np.array(coord2)
    coord1 = np.c_[coord1, np.ones((np.array(coord1)).shape[0])]

    coord1, coord2 = coord1.transpose(), coord2.transpose()
    mat = np.matmul(coord2, np.linalg.pinv(coord1))
    dst = cv2.warpAffine(src, mat, (size[0], size[1]), flags=cv2.INTER_LINEAR)

    return dst

