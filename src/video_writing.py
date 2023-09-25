import open3d as o3d
import numpy as np
import cv2
import glob
import tqdm

from lidar_to_camera_projection import LiDARtoCamera

if __name__ == "__main__":
    ## Read all calib files, images and point cloud files
    calib_files = sorted(glob.glob("../data/calib/*.txt"))
    scenario_images = sorted(glob.glob("../data/scenario/images/*.png"))
    scenario_points = sorted(glob.glob("../data/scenario/points/*.pcd"))
    image = cv2.imread(scenario_images[0])

    ## Create a LiDAR to Camera object
    lidar2camera = LiDARtoCamera(calib_files[0])

    ## Create a video writer object
    output_handle = cv2.VideoWriter("../output/3d_to_2d_projection.avi",
                                    cv2.VideoWriter_fourcc(*'DIVX'),
                                    fps=15,
                                    frameSize=(image.shape[1], image.shape[0]))
    
    ## Create a progress bar
    pbar = tqdm.tqdm(total=len(scenario_images), position=0, leave=True)
    
    for im, pcd in zip(scenario_images, scenario_points):
        image = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
        points = np.asarray(o3d.io.read_point_cloud(pcd).points)
        processed_image = lidar2camera.show_pcd_on_image(image, points)
        output_handle.write(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        pbar.update(1)

    output_handle.release()

