import cv2

def delaunay(img, landmarks):

    subdiv = cv2.Subdiv2D((0, 0, img.shape[1], img.shape[0]))

    for item in landmarks:
        subdiv.insert(item)

    return (subdiv.getTriangleList())

def delaunay2(img, triangleIndex, landmarks, delaunay_color):

    for j in range(len(triangleIndex)):

        x, y, z = triangleIndex[j][0], triangleIndex[j][1], triangleIndex[j][2]
        pt1, pt2, pt3 = landmarks[x], landmarks[y], landmarks[z]

        cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

    return img
