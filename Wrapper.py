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
from codes.api import PRN
from codes.Face_Swap import face_swap
from codes.PRNet import PRNet_process
import os





def main():


    Parser = argparse.ArgumentParser()
    Parser.add_argument('--video', default='', help='The path to a video to do the face swap')
    Parser.add_argument('--image', default='Rambo.jpg', help='the path to an image you want to replace in a video')
    Parser.add_argument('--method', default='tri', help='swap method. Choose between tps, tri, and prnet')
    Parser.add_argument('--image_swap', default='',  help='set to True if you like to swap two arbitrary images '
                                                             '(you should then set --src_image and --dst_image paths)')
    Parser.add_argument('--triangulation', default='', help='path to the image you want to plot the '
                                                                        'landmarks and triangulation')
    Parser.add_argument('--src_image', default='hermione.jpg', help='the path to an image you want to replace to another '
                                                                 'image')
    Parser.add_argument('--dst_image', default='ron.jpg', help='the path to an image you want to swap with source '
                                                                    'image')
    Parser.add_argument('--fps', default=30, type=int,  help='frame per second')
    Parser.add_argument('--tf', default='', help='set to True if you want to swap two faces within video')
    Parser.add_argument('--name', default='Output', type=str, help='Output video name')








    args = Parser.parse_args()
    fs = face_swap()
    prn = None
    if args.method == 'prnet':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        prn = PRN(is_dlib=True)

    if args.triangulation:
        src_image = cv2.imread(args.triangulation)
        fs.Delaunay_triangulation(src_image)

    if args.image_swap:
        dst_image = cv2.imread(args.dst_image)
        src_image = cv2.imread(args.src_image)

        if args.method =='tri':
            fs.DLN_swap(src_image, dst_image, plot=True)
        elif args.method =='tps':
            fs.TPS_swap(dst_image, src_image, plot=True)
        elif args.method == 'prnet':
            _, _ = PRNet_process(prn, dst_image, src_image, two_faces=False, plot=True)




    if args.video:
        image = cv2.imread(args.image)
        cap = cv2.VideoCapture(args.video)
        frame_index = 0

        _, video_image = cap.read()
        height = video_image.shape[0]
        width = video_image.shape[1]
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(args.name + '.avi', fourcc, args.fps, (width, height))


        while True:
            _, video_image = cap.read()

            operation, new_image = fs.swap_operation(args, copy.deepcopy(video_image), image, prn=prn)
            cv2.imwrite('./video_results/frame_' + str(frame_index) + '.jpg', new_image)
            print('frame ' + str(frame_index) + ' has been saved successfully')
            out.write(new_image)
            frame_index += 1
            if not operation:
                continue
            key = cv2.waitKey(5) & 0xFF
            if key == ord('c'):
                cv2.destroyAllWindows()
                break
        cap.release()

    else:
        print('Nothing to do!')


if __name__ == '__main__':
    main()
