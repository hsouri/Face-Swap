"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project2: FaceSwap

Author:

Hossein Souri (hsouri@umiacs.umd.edu)
PhD Student in Computer Vision and Machine Learning
University of Maryland, College Park

cite:
https://github.com/italojs/facial-landmarks-recognition
https://github.com/wuhuikai/FaceSwap/blob/master/face_swap.py
https://cmsc733.github.io/2019/proj/p2-results/
https://github.com/YadiraF/PRNet

"""

import cv2
import imutils
import numpy as np
from scipy.spatial import Delaunay
from scipy import interpolate
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
import sys
import math
import argparse
import copy
from codes.PRNet import PRNet_process


class face_swap:

    def __init__(self):
        self.path = "codes/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.path)
        self.edge = 40


    def get_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def rand_color(self):
        return  (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

    def landmarks_detection(self, image, plot=True):
        gray = self.get_gray(image)

        rects = self.detector(gray, 1)

        if not rects.__len__() :
            return None, None, None
        shapes = []
        points = []

        for (i, rect) in enumerate(rects):
            point = []
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            shapes.append(shape)

            for (x, y) in shape:
                point.append((x, y))

            if plot:
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 3, self.rand_color(), -1)

            points.append(point)
        if plot:
            cv2.imwrite("Landmarks.png", image)

        return True, points, shapes

    def draw_delaunay(self, img, subdiv, plot=True):

        triangleList = subdiv.getTriangleList()

        for t in triangleList:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            cv2.line(img, pt1, pt2, self.rand_color(), 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, self.rand_color(), 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, self.rand_color(), 1, cv2.LINE_AA, 0)

        if plot:
            cv2.imwrite("Triangulation.png", img)


    def Delaunay_triangulation(self, image, plot=True):
        size = image.shape
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect)
        check, points , shapes = self.landmarks_detection(image, plot=True)
        for point in points[0]:
            subdiv.insert(point)

        if plot:
            self.draw_delaunay(image.copy(), subdiv)

    def bounding_box(self, image, shapes):
        upper_left_x, upper_left_y = np.min(shapes, 0)
        lower_right_x, lower_right_y = np.max(shapes, 0)
        shift = np.array([[upper_left_x, upper_left_y]])
        box = (upper_left_x - 1, upper_left_y -1, lower_right_x + 1, lower_right_y + 1)

        return box, shapes - shift, image[box[1]:box[3], box[0]:box[2]]


    def bilinear_interpolate(self, img, coords):
        """ Interpolates over every image channel
        http://en.wikipedia.org/wiki/Bilinear_interpolation
        :param img: max 3 channel image
        :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
        :returns: array of interpolated pixels with same shape as coords
        """
        int_coords = np.int32(coords)
        x0, y0 = int_coords
        dx, dy = coords - int_coords

        # 4 Neighour pixels
        q11 = img[y0, x0]
        q21 = img[y0, x0 + 1]
        q12 = img[y0 + 1, x0]
        q22 = img[y0 + 1, x0 + 1]

        btm = q21.T * dx + q11.T * (1 - dx)
        top = q22.T * dx + q12.T * (1 - dx)
        inter_pixel = top * dy + btm * (1 - dy)

        return inter_pixel.T


    def Barycentric(self, tri, tri_area, src_features, dst_features):
        A_delta = np.vstack((src_features[tri].T, [1, 1, 1]))
        B_delta = np.vstack((dst_features[tri].T, [1, 1, 1]))
        x, y, z = A_delta @ np.linalg.inv(B_delta) @ np.vstack((tri_area.T, np.ones(tri_area.__len__())))
        return np.array([x, y])



    def get_box(self, features):
        box = []

        for col in range(np.min(features[:, 1]), np.max(features[:, 1]) + 1):
            for row in range(np.min(features[:, 0]), np.max(features[:, 0]) + 1):
                box.append((row, col))

        return np.asarray(box, dtype=np.uint32)

    def mask_from_points(self,size, points, erode_flag=1):
        radius = 10  # kernel size
        kernel = np.ones((radius, radius), np.uint8)

        mask = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
        if erode_flag:
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def mask_from_points(self,size, points, erode_flag=1):
        radius = 10  # kernel size
        kernel = np.ones((radius, radius), np.uint8)

        mask = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
        if erode_flag:
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    def apply_mask(self,img, mask):

        """ Apply mask to supplied image
        :param img: max 3 channel image
        :param mask: [0-255] values in mask
        :returns: new image with mask applied
        """
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        return masked_img

    def correct_colours(self,im1, im2, landmarks1):
        COLOUR_CORRECT_BLUR_FRAC = 0.75
        LEFT_EYE_POINTS = list(range(42, 48))
        RIGHT_EYE_POINTS = list(range(36, 42))

        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur = im2_blur.astype(int)
        im2_blur += 128 * (im2_blur <= 1)

        result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result



    def blending(self, dst_points, dst_face, warped_src_face):
        h, w = dst_face.shape[:2]
        ## Mask for blending
        mask = self.mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask * mask_src, dtype=np.uint8)
        ## Correct color
        # if args.correct_color:
        warped_src_face = self.apply_mask(warped_src_face, mask)
        dst_face_masked = self.apply_mask(dst_face, mask)
        warped_src_face = self.correct_colours(dst_face_masked, warped_src_face, dst_points)

        ## Shrink the mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        ##Poisson Blending
        r = cv2.boundingRect(mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)


        return output


    def replace(self, dst_image, box, blended_source):
        dst_image_copy = dst_image.copy()
        dst_image_copy[box[1]:box[3], box[0]:box[2]] = blended_source
        return dst_image_copy

    def DLN_swap(self, src_image, dst_image, plot=False):

        check, points, shapes = self.landmarks_detection(src_image, plot=False)
        if not check:
            return None, None
        src_box_coordinates, src_features, src_cropped = self.bounding_box(src_image, shapes[0])

        check, points, shapes = self.landmarks_detection(dst_image, plot=False)
        if not check:
            return None, None
        dst_box_coordinates, dst_features, dst_cropped = self.bounding_box(dst_image, shapes[0])




        triangles = Delaunay(dst_features)

        dst_box = self.get_box(dst_features)

        box_inices = triangles.find_simplex(dst_box)

        warped_image = np.zeros((dst_cropped.shape[0], dst_cropped.shape[1], 3), dtype=np.uint8)

        for tri_index, tri in enumerate(triangles.simplices):
            tri_area = dst_box[box_inices == tri_index]
            x, y = tri_area.T

            barycentric = self.Barycentric(tri, tri_area, src_features, dst_features)
            warped_image[y, x] = self.bilinear_interpolate(src_cropped, barycentric)

        if plot:
            self.save_image('warped', warped_image)

        blended_src_image = self.blending(dst_features, dst_cropped, warped_image)

        des_swaped_image = self.replace(dst_image, dst_box_coordinates, blended_src_image)

        if plot:
            # self.save_image('warped', warped_image)
            # self.save_image('blended', blended_src_image)
            self.save_image('tri_swaped_image', des_swaped_image)

        return True, des_swaped_image


    def U(self, r):
        return (r ** 2) * (math.log(r ** 2))

    def lp_norm(self, p1, p2, order=2):

        if np.linalg.norm((-p1 + p2), ord=order) == 0:
            return sys.float_info.epsilon
        return np.linalg.norm((-p1 + p2), ord=order)


    def get_K(self, features):
        len = features.__len__()
        K = [[self.U(self.lp_norm(features[i], features[j], order=2)) for i in range(len)] for j in range(len)]
        return np.array(K)

    def get_fat_mat(self, K, P):
        fat = np.concatenate((np.concatenate((K, P), axis=1), np.concatenate((P.T, np.zeros([3, 3])), axis=1)), axis=0)
        return fat


    def TPS_weights(self, src_features, dst_features, _lambda=sys.float_info.epsilon):
        _lambda = 200
        num_points = src_features.__len__()
        P = np.concatenate((src_features, np.ones((num_points, 1))), axis=1)
        K = self.get_K(src_features)
        Inverse = np.linalg.inv(self.get_fat_mat(K, P) + _lambda * np.identity(num_points + 3))
        v = np.concatenate((dst_features, np.array([0, 0, 0])), axis=0)
        w = Inverse @ v
        return w

    def f(self, point, features, w):
        K = np.zeros([features.shape[0], 1])
        for i in range(features.shape[0]):
            K[i] = self.U(self.lp_norm(features[i], point, order=2))
        f = w[-1] + w[-3] * point[0] + w[-2] * point[1] + K.T @ w[0:-3]
        return f

    def get_coordinates(self, features):
        grid_x = np.arange(min(features[:, 0]), max(features[:, 0]))
        grid_y = np.arange(min(features[:, 1]), max(features[:, 1]))
        row, col = np.mgrid[grid_x[0]: grid_x[-1] + 1, grid_y[0]: grid_y[-1] + 1]
        return np.vstack((row.ravel(), col.ravel())).T


    def get_color(self, src_features, min , max, coords, w):
        colors = np.zeros_like(coords[:, 0])
        for i in range(coords.shape[0]):
            colors[i] = self.f(coords[i, :], src_features, w)
        colors[colors < min] = min
        colors[colors > max] = max
        return colors


    def TPS_blend(self,src_image, src_warped_img, masked_warped_img, plot=False):
        r = cv2.boundingRect(masked_warped_img)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        blended_image = cv2.seamlessClone(src_warped_img.copy(), src_image, masked_warped_img, center, cv2.NORMAL_CLONE)
        if plot:
            self.save_image("tps_swaped_image", blended_image)

        return blended_image

    def TPS_swap(self, src_image, dst_image, plot=False):
        check1, _, src_shapes = self.landmarks_detection(src_image, plot=False)
        # src_box_coordinates, src_features, src_cropped = self.bounding_box(src_image, shapes[0])

        check2, _, dst_shapes = self.landmarks_detection(dst_image, plot=False)
        # dst_box_coordinates, dst_features, dst_cropped = self.bounding_box(dst_image, shapes[0])

        if (not check1) or (not check2):
            return None, None

        w_x = self.TPS_weights(src_shapes[0], dst_shapes[0][:, 0])
        w_y = self.TPS_weights(src_shapes[0], dst_shapes[0][:, 1])
        coords = self.get_coordinates(src_shapes[0])
        color_x = self.get_color(src_shapes[0], min(dst_shapes[0][:, 0]), max(dst_shapes[0][:, 0]), coords, w_x)
        color_y = self.get_color(src_shapes[0], min(dst_shapes[0][:, 1]), max(dst_shapes[0][:, 1]), coords, w_y)

        w, h = dst_image.shape[:2]
        # ## Mask for blending
        mask = self.mask_from_points((w, h), dst_shapes[0])
        src_warped_img = src_image.copy()
        masked_warped_img = np.zeros_like(src_warped_img[:, :, 0])

        for index in range(color_x.shape[0]):
            if mask[color_y[index], color_x[index]] > 0:
                src_warped_img[coords[index, 1], coords[index, 0], :] = dst_image[color_y[index], color_x[index], :]
                masked_warped_img[coords[index, 1], coords[index, 0]] = 255

        if plot:
            self.save_image('tps_warped', src_warped_img)



        return True, self.TPS_blend(src_image, src_warped_img, masked_warped_img, plot=plot)


    def save_image(self, name, image):
        cv2.imwrite(name + ".png", image)

    def get_face(self, image, rect, edge=25):
        return image[rect.top() - edge: rect.bottom() + edge, rect.left() - edge: rect.right() + edge, :]




    def get_images(self, rects, video_image, image, edge=25):

        if len(rects) == 1:
            src_image = self.get_face(video_image, rects[0], edge=edge)
            dst_image = image
            return src_image, dst_image

        elif len(rects)>1:
            src_image = self.get_face(video_image, rects[0], edge=edge)
            dst_image = self.get_face(video_image, rects[1], edge=edge)
            return copy.deepcopy(src_image), copy.deepcopy(dst_image)



    def resize(self, image, rect, edge=25):
        return cv2.resize(image,
                   ((rect.right() + edge) - (rect.left() - edge), (rect.bottom() + edge) - (rect.top() - edge)))

    def replace_output(self, image, output, rect, edge=25):
        image[rect.top() - edge:rect.bottom() + edge, rect.left() - edge:rect.right() + edge, :] = output


    def swap_operation(self, args, video_image, image, prn=None):


        if args.method == 'prnet':

            ckeck, output = PRNet_process(prn, video_image, image, args.tf)
            if ckeck is None:
                return None, video_image
            return True, output


        edge = self.edge
        rects = self.detector(video_image, 1)
        if not rects.__len__():
            return None, video_image

        copy_video_image = copy.deepcopy(video_image)
        src_image, dst_image = self.get_images(rects, copy_video_image, image, edge=edge)


        if args.method == 'tps':

            check, output1 = self.TPS_swap(src_image, dst_image)
            if not check:
                return None, video_image
            output1 = self.resize(output1, rects[0], edge=edge)
            self.replace_output(video_image, output1, rects[0], edge=edge)

            if len(rects) > 1:
                check, output2 = self.TPS_swap(dst_image, src_image)
                if not check:
                    return None, video_image
                output2 = self.resize(output2, rects[1], edge=edge)
                self.replace_output(video_image, output2, rects[1], edge=edge)


        elif args.method =='tri':

            ckeck, output1 = self.DLN_swap(dst_image, src_image)
            if not ckeck:
                return None, video_image
            output1 = self.resize(output1, rects[0], edge=edge)
            self.replace_output(video_image, output1, rects[0], edge=edge)
            if len(rects) > 1:
                check, output2 = self.DLN_swap(src_image, dst_image)
                if not ckeck:
                    return None, video_image
                output2 = self.resize(output2, rects[1], edge=edge)
                self.replace_output(video_image, output2, rects[1], edge=edge)




        return True, video_image