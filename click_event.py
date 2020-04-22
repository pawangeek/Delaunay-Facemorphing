import cv2

def click(img):

    h, w, c = img.shape
    landmarks = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            landmarks.append((int(x), int(y)))

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', onMouse)
    cv2.waitKey(0)

    landmarks.extend([(0,0), (0,h-1), (w-1,h-1), (w-1,0)])
    return landmarks
