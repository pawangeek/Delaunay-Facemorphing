import dlib
import random

from cv2 import imread, getAffineTransform, resize, warpAffine, Subdiv2D,\
    boundingRect, fillConvexPoly, INTER_LINEAR, BORDER_REFLECT_101, waitKey, VideoWriter_fourcc, VideoWriter

import numpy as np

FILE_SAVE_ERROR = 2
MODEL_ERROR = 1
FACE_FEATURES_MODEL_PATH = "predictor.dat"


def index_find(point, landmarks):

    for index in range(len(landmarks)):
        if point == landmarks[index]:
            return index


def make_correspondence(predictor_path, first_input_image, second_input_image):
    """
    Uses a dlib trained model to predict the coordinates features in input faces
    input : a path to a dlib trained model and two face images
    output: two sets of points corresponding to the location of features
    like eyes and noses in the input images
    """
    detector = dlib.get_frontal_face_detector()
    predictor_model = dlib.shape_predictor(predictor_path)
    # Setting up some initial values.

    list_of_images = [first_input_image, second_input_image]
    first_image_face_features_points = []
    second_image_features_points = []
    j = 1
    max_number_of_learned_points = 68

    for img in list_of_images:
        size = (img.shape[0], img.shape[1])
        if j == 1:
            current_feature_points = first_image_face_features_points
        else:
            current_feature_points = second_image_features_points

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        # Also give error if face is not found.
        dets = detector(img, 1)
        j += 1

        for k, d in enumerate(dets):
            # Get the landmarks/parts for the face in box d.
            shape = predictor_model(img, d)
            for i in range(0, max_number_of_learned_points):
                current_feature_points.append((int(shape.part(i).x),
                                               int(shape.part(i).y)))
            current_feature_points.append((1, 1))
            current_feature_points.append((size[1]-1, 1))
            current_feature_points.append(((size[1]-1)//2, 1))
            current_feature_points.append((1, size[0]-1))
            current_feature_points.append((1, (size[0]-1)//2))
            current_feature_points.append(((size[1]-1)//2, size[0]-1))
            current_feature_points.append((size[1]-1, size[0]-1))
            current_feature_points.append(((size[1]-1)//2, (size[0]-1)//2))

    return [first_image_face_features_points, second_image_features_points]


def rect_contains(rectangle, point):
    """
    checks if a point is contained within the bounds
    of a rectangle plane
    """
    if point[0] < rectangle[0]:
        return False
    elif point[1] < rectangle[1]:
        return False
    elif point[0] > rectangle[2]:
        return False
    elif point[1] > rectangle[3]:
        return False
    return True


# def draw_delaunay(triangleList,  rectangle):
#     """
#     given a subdivision, a dictionary of points
#     and a rectangle, use a brute force method to find and return
#     appoints fall under delaunay triangulation
#     constraints
#     """
#     triangulation_points = []
#     for final_triangle in triangleList:
#
#         point_1 = (int(final_triangle[0]), int(final_triangle[1]))
#         point_2 = (int(final_triangle[2]), int(final_triangle[3]))
#         point_3 = (int(final_triangle[4]), int(final_triangle[5]))
#         if rect_contains(rectangle, point_1) and rect_contains(rectangle, point_2) and rect_contains(rectangle, point_3):
#             triangulation_points.append((dictionary[point_1], dictionary[point_2], dictionary[point_3]))
#
#     return triangulation_points


def make_delaunay(image, list_of_points):
    """
    the input is an image file read in by opencv and a list of average points
    the output is the final triangulation points in a list of tuples
    """
    size = image.shape
    rectangle = (0, 0, size[1], size[0])
    # Create an instance of Subdiv2D.
    subdiv = Subdiv2D(rectangle)
    # Make a points list and a searchable dictionary.
    points = []  # making a tuple of points

    for item in list_of_points:
        points.append(item)

    # Insert points into subdiv
    for point in points:
        subdiv.insert(point)

    triangleList = subdiv.getTriangleList()

    # Make a delaunay triangulation list.
    return triangleList


def apply_affine_transform(src, src_triangle, dst_triangle, size):
    """
    calculates an affine matrix using input triangles and uses that matrix
    to apply an affine transform to the input image to produce a final size big image
    """
    warp_mat = getAffineTransform(np.float32(src_triangle), np.float32(dst_triangle))

    # Apply the Affine Transform just found to the src image
    dst = warpAffine(src, warp_mat, (size[0], size[1]), None, flags=INTER_LINEAR, borderMode=BORDER_REFLECT_101)

    return dst


def morph_triangle(image_1, image_2, img, triangle_1, triangle_2, final_triangle, morphing_rate):
    """
    given three triangles, a morphing rate, and two images, find the corresponding
    triangular regions wrap them around the input images and blend the result into the
    the final image according to the morphing rate
    """

    # Find bounding rectangle for each triangle
    rectangle_1 = boundingRect(np.float32([triangle_1]))
    rectangle_2 = boundingRect(np.float32([triangle_2]))
    rectangle = boundingRect(np.float32([final_triangle]))

    # Offset points by left top corner of the respective rectangles
    triangle_1_rect = []
    triangle_2_rect = []
    final_triangle_rect = []

    for i in range(0, 3):
        final_triangle_rect.append(((final_triangle[i][0] - rectangle[0]), (final_triangle[i][1] - rectangle[1])))
        triangle_1_rect.append(((triangle_1[i][0] - rectangle_1[0]), (triangle_1[i][1] - rectangle_1[1])))
        triangle_2_rect.append(((triangle_2[i][0] - rectangle_2[0]), (triangle_2[i][1] - rectangle_2[1])))

    # Get mask by filling triangle
    mask = np.zeros((rectangle[3], rectangle[2], 3), dtype=np.float32)
    fillConvexPoly(img=mask, points=np.int32(final_triangle_rect), color=(1.0, 1.0, 1.0), lineType=16, shift=0)

    # Apply warpImage to small rectangular patches
    img_1_rectangle = image_1[rectangle_1[1]:rectangle_1[1] + rectangle_1[3], rectangle_1[0]:rectangle_1[0] + rectangle_1[2]]
    img_2_rectangle = image_2[rectangle_2[1]:rectangle_2[1] + rectangle_2[3], rectangle_2[0]:rectangle_2[0] + rectangle_2[2]]

    size = (rectangle[2], rectangle[3])
    warp_image_1 = apply_affine_transform(img_1_rectangle, triangle_1_rect, final_triangle_rect, size)
    warap_image_2 = apply_affine_transform(img_2_rectangle, triangle_2_rect, final_triangle_rect, size)

    # Alpha blend rectangular patches
    img_rect = (1.0 - morphing_rate) * warp_image_1 + morphing_rate * warap_image_2

    # Copy triangular region of the rectangular patch to the output image
    img[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]] = \
        img[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]] *\
        (1 - mask) + img_rect * mask


def make_morph(image_filename_1, image_filename_2, fps):
    """
    using all the previous helper function and the pretained morphing model
    to morph two input images into a final morphed image
    and  save it in the directory where this file is currently sitting
    """
    image_1 = imread(image_filename_1)
    image_2 = imread(image_filename_2)

    size_1 = image_1.shape
    size_2 = image_2.shape

    height = min(size_1[1], size_2[1])
    width = min(size_1[0], size_2[0])

    final_size = (width, height)

    img_1 = resize(image_1, final_size)
    img_2 = resize(image_2, final_size)

    features = make_correspondence(FACE_FEATURES_MODEL_PATH, img_1, img_2)

    first_image_points = features[0]
    second_image_points = features[1]

    if len(first_image_points) == 0 or len(second_image_points) == 0:
        return -MODEL_ERROR

    ori_delaunay = make_delaunay(img_1,first_image_points)

    tri_index = []
    for t in ori_delaunay:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        add = [(index_find(pt1, first_image_points), index_find(pt2, first_image_points), index_find(pt3, first_image_points))]
        tri_index.extend(add)

    fourcc = VideoWriter_fourcc(*"X264")
    second = 5
    frames = fps*second
    n = random.randint(0, 10**8)
    videoWriter = VideoWriter(str(n)+".mkv", fourcc, fps, final_size)

    for k in range(0, frames + 1):
        alpha = k / frames
        landmarks_Middle = []
        # Face feature point interpolation
        for i in range(len(first_image_points)):
            x = int((1 - alpha) * first_image_points[i][0] + alpha * second_image_points[i][0])
            y = int((1 - alpha) * first_image_points[i][1] + alpha * second_image_points[i][1])
            landmarks_Middle.append((x, y))

        imgMorph = np.zeros(img_1.shape, dtype=img_1.dtype)

        for j in range(len(tri_index)):
            # Get point index
            x = tri_index[j][0]
            y = tri_index[j][1]
            z = tri_index[j][2]

            # Get the point coordinates according to the index
            t1 = [first_image_points[x], first_image_points[y], first_image_points[z]]
            t2 = [second_image_points[x], second_image_points[y], second_image_points[z]]
            t = [landmarks_Middle[x], landmarks_Middle[y], landmarks_Middle[z]]

            # Morphing a triangle
            morph_triangle(img_1, img_2, imgMorph, t1, t2, t, alpha)

        videoWriter.write(imgMorph)
        waitKey(10)

    videoWriter.release()
    return str(n)+".mkv"