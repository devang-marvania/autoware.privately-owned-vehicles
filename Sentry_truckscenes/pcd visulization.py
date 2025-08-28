import open3d as o3d
pcd = o3d.io.read_point_cloud("D:\Git Repos\truckscenes data\data\man-truckscenes\samples\RADAR_LEFT_BACK\RADAR_LEFT_BACK_1692868171703928.pcd")  # or .ply, .xyz
o3d.visualization.draw_geometries([pcd])