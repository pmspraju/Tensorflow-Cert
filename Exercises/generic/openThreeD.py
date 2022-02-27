import os
import open3d as o3d
print(o3d.__version__)
o3d.visualization.webrtc_server.enable_webrtc()
if __name__ == "__main__":
    # sample_ply_data = o3d.data.PLYPointCloud()
    # pcd = o3d.io.read_point_cloud(sample_ply_data.path)
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
    path = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Data\pointcloud'
    pcdfile = os.path.join(path,'s2.pts')
    pcd = o3d.io.read_point_cloud(pcdfile,format='pts')
    o3d.visualization.draw_geometries([pcd])