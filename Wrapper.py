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
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def initialize_face_swap(args):
    fs = face_swap()
    prn = None
    if args.method == 'prnet':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        prn = PRN(is_dlib=True)
    return fs, prn

def validate_arguments(args):
    # Validate paths
    if args.image_swap:
        if not os.path.exists(args.src_image):
            logging.error(f"Source image '{args.src_image}' not found.")
            sys.exit(1)
        if not os.path.exists(args.dst_image):
            logging.error(f"Destination image '{args.dst_image}' not found.")
            sys.exit(1)

    if args.video:
        if not os.path.exists(args.video):
            logging.error(f"Video file '{args.video}' not found.")
            sys.exit(1)

    if args.image and not os.path.exists(args.image):
        logging.error(f"Image file '{args.image}' not found.")
        sys.exit(1)

def swap_faces_in_video(args, fs, prn):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logging.error(f"Failed to open video file '{args.video}'.")
        sys.exit(1)
    
    # Read the first frame to get dimensions
    ret, video_image = cap.read()
    if not ret:
        logging.error("Failed to read video frame.")
        sys.exit(1)
    
    height, width = video_image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file_name = f"{args.name}_{timestamp}.avi"
    out = cv2.VideoWriter(out_file_name, fourcc, args.fps, (width, height))

    frame_index = 0
    while True:
        ret, video_image = cap.read()
        if not ret:
            break

        operation, new_image = fs.swap_operation(args, copy.deepcopy(video_image), args.image, prn=prn)
        if operation:
            cv2.imwrite(f'./video_results/frame_{frame_index}.jpg', new_image)
            logging.info(f'Frame {frame_index} has been saved successfully.')
            out.write(new_image)
        
        frame_index += 1

    cap.release()
    out.release()

def swap_faces_in_images(args, fs, prn):
    src_image = cv2.imread(args.src_image)
    dst_image = cv2.imread(args.dst_image)

    if args.method == 'tri':
        fs.DLN_swap(src_image, dst_image, plot=True)
    elif args.method == 'tps':
        fs.TPS_swap(dst_image, src_image, plot=True)
    elif args.method == 'prnet':
        _, _ = PRNet_process(prn, dst_image, src_image, two_faces=False, plot=True)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='', help='The path to a video to do the face swap')
    parser.add_argument('--image', default='Rambo.jpg', help='the path to an image you want to replace in a video')
    parser.add_argument('--method', default='tri', help='swap method. Choose between tps, tri, and prnet')
    parser.add_argument('--image_swap', default='', help='set to True if you like to swap two arbitrary images '
                                                          '(you should then set --src_image and --dst_image paths)')
    parser.add_argument('--triangulation', default='', help='path to the image you want to plot the '
                                                            'landmarks and triangulation')
    parser.add_argument('--src_image', default='hermione.jpg', help='the path to an image you want to replace to another '
                                                                     'image')
    parser.add_argument('--dst_image', default='ron.jpg', help='the path to an image you want to swap with source '
                                                                'image')
    parser.add_argument('--fps', default=30, type=int, help='frame per second')
    parser.add_argument('--tf', default='', help='set to True if you want to swap two faces within video')
    parser.add_argument('--name', default='Output', type=str, help='Output video name')

    args = parser.parse_args()

    # Validate input arguments
    validate_arguments(args)

    # Initialize face swap
    fs, prn = initialize_face_swap(args)

    # Triangulation option
    if args.triangulation:
        src_image = cv2.imread(args.triangulation)
        fs.Delaunay_triangulation(src_image)

    # Image swap option
    if args.image_swap:
        swap_faces_in_images(args, fs, prn)

    # Video processing option
    elif args.video:
        swap_faces_in_video(args, fs, prn)

    else:
        logging.info('Nothing to do!')

if __name__ == '__main__':
    main()
