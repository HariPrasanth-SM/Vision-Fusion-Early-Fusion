import numpy as np
import open3d as o3d

def bin_to_pcd(binfile, pcdfile, reflectivity=True):
    """
    This function convert the point cloud file from binary format to ASCII format

    :param binfile: string, the location of the binfile
    :param pcdfile: string, the location where the pcd file save
    :param reflectivity: bool, whether the reflectivity data is in the bin file or not
    :return True: if the conversion is success, this function will return true
    """
    binfile = np.fromfile(binfile, dtype=np.float32)
    print(binfile.shape)
    binfile = binfile.reshape((-1, 4))

    ## Creating a pointcloud object
    pcd = o3d.geometry.PointCloud()

    ## Updating the points and reflectivity to the pcd object
    pcd.points = o3d.utility.Vector3dVector(binfile[:, :3])

    ## Updating the reflectivity property to a np array
    if reflectivity:
        reflectivity = np.zeros((binfile.shape[0], 3))
        reflectivity[:, 0] = binfile[:, 3]
        pcd.colors = o3d.utility.Vector3dVector(reflectivity)

    ## Write thr pcd file
    o3d.io.write_point_cloud(pcdfile, pcd)
    
    return True

if __name__ == "__main__":
    file_to_open = "../data/velodyne/000031.bin"
    file_to_save = "../data/velodyne/000031.pcd"

    if bin_to_pcd(file_to_open, file_to_save, reflectivity=False):
        print("Bin to PCD conversion done")
        pcd = o3d.io.read_point_cloud(file_to_save)
        o3d.visualization.draw_geometries([pcd])