import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from yolov4.tf import YOLOv4

from lidar_to_camera_projection import LiDARtoCamera
from object_detection_2d import run_obstacle_detection

yolo = YOLOv4(tiny=False)
yolo.classes = "../data/Yolov4/coco.names"
yolo.make_model()
yolo.load_weights("../data/Yolov4/yolov4.weights", weights_type="yolo")

if __name__ == "__main__":
    idx = 0

    ## Load the list of files 
    calib_files = sorted(glob.glob("../data/calib/*.txt"))
    pointcloud_files = sorted(glob.glob("../data/velodyne/*.pcd"))
    image_files = sorted(glob.glob("../data/img/*.png"))
    
    ## Read the image file
    image = cv2.imread(image_files[idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ## Read Point cloud files
    pcd = o3d.io.read_point_cloud(pointcloud_files[idx])
    points_pcd = np.asarray(pcd.points)

    ## Convert from LiDAR to Camera coord
    lidar2cam = LiDARtoCamera(calib_files[idx])

    ## Point cloud data in image
    image_pcd = lidar2cam.show_pcd_on_image(image.copy(), points_pcd)

    ## Object detection 2d
    image_object_detection, pred_bboxes = run_obstacle_detection(image)

    lidar_img_with_bboxes= yolo.draw_bboxes(image_pcd, pred_bboxes)
    fig_fusion = plt.figure(figsize=(14, 7))
    ax_fusion = fig_fusion.subplots()
    ax_fusion.imshow(lidar_img_with_bboxes)
    cv2.imwrite("../output/fig_fusion.png", lidar_img_with_bboxes)
    plt.show()



    
