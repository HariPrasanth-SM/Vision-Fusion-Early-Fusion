import open3d as o3d
from open3d.visualization.draw_plotly import get_plotly_fig

def visualize_pcd(pcd_list, show=True, save="False"):
    """
    This function uses the plotly library to visualize and save the given pcd objects.
    It shows the immage in web browser and saves the pcd file as jpg image.

    :param pcd_list: list of pcd objects 
    :param show: a bool by default, if it is True it visualize the pcd as image
    :param save: a string, if a value passed, it will save the pcd as image in output folder
    :return fig: the plotly fig object with all pcd objects with some processed view
    """
    fig = get_plotly_fig(pcd_list, width=800, height=533, mesh_show_wireframe=False,
                         point_sample_factor=1, front=(1,1,1), lookat=(1,1,1), up=(1,1,1), zoom=1.0)
    #fig.update_scenes(aspectmode='data')
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-70, 70]),
            yaxis=dict(visible=False, range=[-40, 40]),
            zaxis=dict(visible=False, range=[-5, 1]),
            aspectmode='manual', aspectratio=dict(x=2, y=1, z=0.1),
            camera=dict(
                up=dict(x=0.15, y=0, z=1),
                center=dict(x=0, y=0, z=0.1),
                eye=dict(x=-0.3, y=0, z=0.3)
            )
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        scene_dragmode='orbit'
    )

    if show == True:
        fig.show()

    if save != "False":
        fig.write_image("../output/"+save+".jpg", scale=3)

    return fig

if __name__ == "__main__":
    ## Read point cloud file
    pcd = o3d.io.read_point_cloud("../data/velodyne/000031.pcd")

    ## Visualise point cloud
    o3d.visualization.draw_geometries([pcd])

    visualize_pcd([pcd], save="test")
