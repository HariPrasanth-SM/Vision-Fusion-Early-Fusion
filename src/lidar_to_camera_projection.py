import numpy as np
import open3d as o3d
import glob

class LiDARtoCamera():
    """
    This will read the calibration file, and create the transformation matrix from one
    co-ordinate frame to another one
    """
    def __init__(self, calibfile):
        calib_data = self.read_calibfile(calibfile)
        ## Intrinsics param, the images are belongs to P2
        self.P = calib_data["P2"].reshape((3, 4))
        ## Rotation from reference camera coord to rect camera coord
        self.R0 = calib_data["R0_rect"].reshape((3, 3))
        ## Translation from velodyne coord to reference camera coord
        self.V2C = calib_data["Tr_velo_to_cam"].reshape((3, 4))

    def read_calibfile(self, filepath):
        """
        This function reads the calibration text file, and converts its content into a dictionary
        
        :param filepath: string, the path to the calibration file
        :return data: dict, keys and values of the calibration file
        """
        data = dict()
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if (len(line) == 0):
                    continue
                key, value = line.split(":", 1)

                data[key] = np.array([float(x) for x in value.split()])
        return data
    
    def project_pcd_to_image(self, pcd_points):
        """
        This function converts all the points from velodyne co-ord to image co-ord
        It uses the following formula:
        2-D points = P * R0 * R|t * 3-D points

        :param pcd_points: ndarray, Points (x, y, z) of a point cloud with size (n x 3)
        :return img_points: ndarray, Points (x, y) in image co-ord (n x 2)
        """
        ## Homogeneous conversion of the matrices to compatable shape for matrix multiplication
        ## 2-D points = P   * R0  * R|t * 3-D points
        ## (3xn)      = (3x4) (3x3) (3x4) (3xn) convert into
        ## (3xn)      = (3x4) (4x4) (4x4) (4xn)
        P = self.P 
        R0 = np.vstack([np.hstack([self.R0, [[0.], [0.], [0.]]]), [0., 0., 0., 1.]])
        Rt = np.vstack([self.V2C, [0., 0., 0., 1.]])

        ## Homogeneous convertion of the input points dimension from nx3 to 4xn 
        pcd_points = np.transpose(pcd_points)
        pcd_points = np.vstack([pcd_points, np.ones(pcd_points.shape[1])])

        ## Applying the formula
        img_points = np.dot(P, R0)
        img_points = np.dot(img_points, Rt)
        img_points = np.dot(img_points, pcd_points)

        ## Homogeneous to Euclidean conversion
        img_points = np.transpose(img_points)
        img_points[:, 0] /= img_points[:, 2]
        img_points[:, 1] /= img_points[:, 2]

        return img_points[:, :2]

if __name__ == "__main__":
    ## List of calibration file and select first one
    calib_files = sorted(glob.glob("../data/calib/*.txt"))
    idx = 0
    
    ## Read Point cloud files
    pointcloud_files = sorted(glob.glob("../data/velodyne/*.pcd"))
    pcd = o3d.io.read_point_cloud(pointcloud_files[idx])

    ## Convert from LiDAR to Camera coord
    lidar2cam = LiDARtoCamera(calib_files[idx])

    print(lidar2cam.project_pcd_to_image(np.asarray(pcd.points)[:1, :3]))
    
