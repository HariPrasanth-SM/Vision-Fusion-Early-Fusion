import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import statistics
import random
from yolov4.tf import YOLOv4

from lidar_to_camera_projection import LiDARtoCamera
from object_detection_2d import run_obstacle_detection
from fuse_point_clouds_with_bboxes import FusionLidarCamera

def pipeline(image, point_cloud, calib_file):
    "For a pair of 2 Calibrated Images"
    img = image.copy()

    lidar2cam = FusionLidarCamera(calib_file)
    # Show LidAR on Image
    lidar_img = lidar2cam.show_pcd_on_image(image, np.asarray(point_cloud.points))
    # Run obstacle detection in 2D
    result, pred_bboxes = run_obstacle_detection(img)
    # Fuse Point Clouds & Bounding Boxes
    img_final, _ = lidar2cam.lidar_camera_fusion(pred_bboxes, result)
    return img_final

if __name__ == "__main__":
    image = cv2.cvtColor(cv2.imread("../data/img/000031.png"), cv2.COLOR_BGR2RGB)
    pcd = o3d.io.read_point_cloud("../data/velodyne/000031.pcd")
    calib_file = "../data/calib/000031.txt"

    result_img = pipeline(image, pcd, calib_file)

    fig = plt.figure(figsize=(14, 7))
    ax_keeping = fig.subplots()
    ax_keeping.imshow(result_img)
    cv2.imwrite("../output/result_img.png", result_img)
    plt.show()