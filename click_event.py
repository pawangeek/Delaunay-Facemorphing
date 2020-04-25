import cv2

def click(img):

    # Getting dimensions of image
    h, w, c = img.shape

    # Create a empty list to get mouse clicks
    landmarks = []

    def onMouse(event, x, y, flags, param):

        # cv2.EVENT_LBUTTONDOWN : indicates that the left mouse button is pressed.
        # Docs : https://docs.opencv.org/3.4/d7/dfc/group__highgui.html

        if event == cv2.EVENT_LBUTTONDOWN:

            # Appending coordinates of mouse clicks
            landmarks.append((int(x), int(y)))

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', onMouse)
    cv2.waitKey(0)

    # Adding corner points to our above created list
    landmarks.extend([(0,0), (0,h-1), (w-1,h-1), (w-1,0)])
    return landmarks
