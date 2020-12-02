
"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project2: FaceSwap

Author:

Hossein Souri (hsouri@umiacs.umd.edu)
PhD Student in Computer Vision and Machine Learning
University of Maryland, College Park

cite:

https://github.com/YadiraF/PRNet

"""




import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse

from codes.utils.render import render_texture
import cv2


def PRNet_process(prn, image, ref_image, two_faces, plot=False):


    [h, w, _] = image.shape

    # -- 1. 3d reconstruction -> get texture.

    if two_faces:

        all_poses = prn.process2(image)
        if all_poses == None:
            return None, image

        if len(all_poses) == 2:
            _, output = PRNet_swap(prn, image, image, all_poses[0], all_poses[1], h, w)
            _, output = PRNet_swap(prn, output, image, all_poses[1], all_poses[0], h, w, plot=plot)
        else:
            return None, image

        return all_poses, output

    else:

        pos = prn.process(image)
        if pos is None:
            return pos, image
        ref_pos = prn.process(ref_image)
        _, output = PRNet_swap(prn, image, ref_image, pos, ref_pos, h, w, plot=plot)
        return pos, output



def PRNet_swap(prn, image, ref_image, pos, ref_pos, h, w, plot=False):


    # -- 1. 3d reconstruction -> get texture.

    vertices = prn.get_vertices(pos)
    image = image / 255.
    texture = cv2.remap(image, pos[:, :, :2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0))


    ref_image = ref_image / 255.
    ref_texture = cv2.remap(ref_image, ref_pos[:, :, :2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    ref_vertices = prn.get_vertices(ref_pos)
    new_texture = ref_texture  # (texture + ref_texture)/2.

    # -- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c=1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c=3)
    new_image = image * (1 - face_mask[:, :, np.newaxis]) + new_image * face_mask[:, :, np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask > 0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1]) / 2 + 0.5), int((vis_min[0] + vis_max[0]) / 2 + 0.5))
    output = cv2.seamlessClone((new_image * 255).astype(np.uint8), (image * 255).astype(np.uint8),
                               (face_mask * 255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    if plot:
        cv2.imwrite('PRNet_swaped_image.png', output)

    return pos, output