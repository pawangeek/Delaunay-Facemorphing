from delaunay import delaunay, delaunay2, getAffine
from click_event import click
import cv2
import numpy as np


# Index through the returned triangle point
# To preserve order
def index_find(point, landmarks):
    for index in range(len(landmarks)):
        if point == landmarks[index]:
            return index


def morphTriangle(img1, img2, img, t1, t2, t, alpha):

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect, t2Rect, tRect = [],[],[]
    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))

    mask = np.zeros((r[3], r[2], 3), dtype='float32')
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = getAffine(img1Rect, t1Rect, tRect, size)
    warpImage2 = getAffine(img2Rect, t2Rect, tRect, size)

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1-mask) + imgRect*mask


if __name__ == '__main__':

    # Setting frames and fps for video
    frames, fps = 20, 10

    # Getting images from path and reading them using cs2.imread()
    path1, path2 = "images/george.jpg", "images/bush.jpg"
    img1, img2 = cv2.imread(path1), cv2.imread(path2)

    # Getting control by click on image + predefined corners
    landmarks1 = click(img1)
    landmarks2 = click(img2)

    # Getting triangles by calling delaunay function
    ori_delaunay = delaunay(img1, landmarks1)
    h, w, c = img1.shape

    tri_index = []
    for t in ori_delaunay:
        pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])

        # Creating a combined point of (x,y)
        add = [(index_find(pt1, landmarks1), index_find(pt2, landmarks1), index_find(pt3, landmarks1))]
        tri_index.extend(add)

    # Reference : https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html

    # FourCC is a 4-byte code used to specify the video codec
    # In Windows: DIVX (More to be tested and added) as written in tutorial
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")

    # Docs : https://docs.opencv.org/master/dd/d9e/classcv_1_1VideoWriter.html
    videoWriter = cv2.VideoWriter("Morph.mp4v", fourcc, fps, (w, h))

    for k in range(0, frames + 1):
        alpha, landmarks_Middle = k / frames, []
        # Defining alpha = k/frames implies 1 for last frame and minimum for 1st
        # Create equal partition of features in each frame

        for i in range(len(landmarks1)):
            x = int(((1-alpha) * landmarks1[i][0]) + (alpha * landmarks2[i][0]))
            y = int(((1-alpha) * landmarks1[i][1]) + (alpha * landmarks2[i][1]))
            landmarks_Middle.append((x, y))

        # Creating a image with as same as shape of img1 having zeroes
        imgMorph = np.zeros(img1.shape, dtype=img2.dtype)

        for j in range(len(tri_index)):

            x, y, z = tri_index[j][0], tri_index[j][1], tri_index[j][2]
            t1 = [landmarks1[x], landmarks1[y], landmarks1[z]]
            t2 = [landmarks2[x], landmarks2[y], landmarks2[z]]
            t = [landmarks_Middle[x], landmarks_Middle[y], landmarks_Middle[z]]

            # Doing morphing with passed triangles and images
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

        imgMorph_delaunay = delaunay2(imgMorph, tri_index, landmarks_Middle, (255, 255, 255))

        cv2.imshow("Morphed Face", np.uint8(imgMorph_delaunay))
        cv2.imwrite(str(k)+".jpg", imgMorph_delaunay)
        videoWriter.write(imgMorph_delaunay)
        cv2.waitKey(50)

    videoWriter.release()
