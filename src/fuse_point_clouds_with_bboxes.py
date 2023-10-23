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

yolo = YOLOv4(tiny=False)
yolo.classes = "../data/Yolov4/coco.names"
yolo.make_model()
yolo.load_weights("../data/Yolov4/yolov4.weights", weights_type="yolo")

class FusionLidarCamera(LiDARtoCamera):
    def rectContains(self, rect,pt, w, h, shrink_factor = 0):       
        x1 = int(rect[0]*w - rect[2]*w*0.5*(1-shrink_factor)) # center_x - width /2 * shrink_factor
        y1 = int(rect[1]*h-rect[3]*h*0.5*(1-shrink_factor)) # center_y - height /2 * shrink_factor
        x2 = int(rect[0]*w + rect[2]*w*0.5*(1-shrink_factor)) # center_x + width/2 * shrink_factor
        y2 = int(rect[1]*h+rect[3]*h*0.5*(1-shrink_factor)) # center_y + height/2 * shrink_factor

        return x1 < pt[0]<x2 and y1 <pt[1]<y2 

    def filter_outliers(self, distances):
        inliers = []
        mu  = statistics.mean(distances)
        std = statistics.stdev(distances)
        for x in distances:
            if abs(x-mu) < std:
                inliers.append(x)
        return inliers

    def get_best_distance(self, distances, technique="closest"):
        if technique == "closest":
            return min(distances)
        elif technique =="average":
            return statistics.mean(distances)
        elif technique == "random":
            return random.choice(distances)
        else:
            return statistics.median(sorted(distances))

    def lidar_camera_fusion(self, pred_bboxes, image):
        img_bis = image.copy()

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        distances = []
        for box in pred_bboxes:
            distances = []
            for i in range(self.pcd_img_points.shape[0]):
                #depth = self.imgfov_pc_rect[i, 2]
                depth = self.pcd_points_in_img[i,0]
                if (self.rectContains(box, self.pcd_img_points[i], image.shape[1], image.shape[0], shrink_factor=0.2)==True):
                    distances.append(depth)

                    color = cmap[int(510.0 / depth), :]
                    cv2.circle(img_bis,(int(np.round(self.pcd_img_points[i, 0])), int(np.round(self.pcd_img_points[i, 1]))),2,color=tuple(color),thickness=-1,)
            h, w, _ = img_bis.shape
            if (len(distances)>2):
                distances = self.filter_outliers(distances)
                best_distance = self.get_best_distance(distances, technique="average")
                cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0]*w),int(box[1]*h)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 3, cv2.LINE_AA)    
            distances_to_keep = []
        
        return img_bis, distances

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
    lidar2cam = FusionLidarCamera(calib_files[idx])

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

    final_result, _ = lidar2cam.lidar_camera_fusion(pred_bboxes, image)

    fig_keeping = plt.figure(figsize=(14, 7))
    ax_keeping = fig_keeping.subplots()
    ax_keeping.imshow(final_result)
    cv2.imwrite("../output/final_result.png", final_result)
    plt.show()



    
