# Face-Swap
Python end-to-end pipeline to swap faces in videos and images.


The aim of this code is to implement an end-to-end pipeline to swap faces in a video just like Snapchat’s face swap filter or this face swap website.


# Landmark detection and Triangulation

1- If you want to see triangulation and landmark detection on an arbitrary image run this:


```shell
python Wrapper.py --triangulation [path to the image]
```

# Face swap between two images
Download the PRN trained model at BaiduDrive or GoogleDrive, and put it into codes/Data/net-data

Link: https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view

2- To swap faces of two arbitrary images run this:

```shell
python Wrapper.py --image_swap true --src_image [path to source image] --dst_image [path to destination image] --method [metod key]
```

Note: Method key should be choosen between tri, tps, or prnet.

"tri" stands for Delaunay Triangulation.

"tps" stands for Thin Plate Splines.

"prnet" stands for PR Net.

# Face swap in a video and a target iamge
3- Two swap a face in a video with an target image run this:

```shell
python Wrapper.py --video [path to the .mp4 video] --image [path to destination image] --method [metod key] --name [output video name]
```

Note: You can set frame per second by setting --fps [your desire frame per second]


# Face swap in a single video
4- Two swap two faces in a video run this:

```shell
python Wrapper.py --video [path to the .mp4 video] --tf true --method [metod key] --name [output video name]
```

Note: You can set frame per second by setting --fps [your desire frame per second]
