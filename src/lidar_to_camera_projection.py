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
        self.P = calib_data["P2"]
        ## Rotation from reference camera coord to rect camera coord
        self.R0 = calib_data["R0_rect"]
        ## Translation from velodyne coord to reference camera coord
        self.V2C = calib_data["Tr_velo_to_cam"]

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

if __name__ == "__main__":
    ## List of calibration file and select first one
    calibfiles = sorted(glob.glob("../data/calib/*txt"))
    idx = 0
    
    ## Convert from LiDAR to Camera coord
    lidar2cam = LiDARtoCamera(calibfiles[idx])
