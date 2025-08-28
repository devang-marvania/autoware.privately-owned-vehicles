
# Import necessary libraries
import matplotlib.pyplot as plt  # For plotting and saving images
from truckscenes import TruckScenes  # For loading and handling truck scenes dataset
from pathlib import Path  # For handling file and folder paths

import os
import os.path as osp

from datetime import datetime
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from PIL import Image
from matplotlib import cm, rcParams
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.cm import ScalarMappable
from pyquaternion import Quaternion

from truckscenes.utils import colormap
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from truckscenes.utils.geometry_utils import view_points, transform_matrix, \
    BoxVisibility
from truckscenes.utils.visualization_utils import TruckScenesExplorer

class MyTruckScenesExplorer(TruckScenesExplorer):
    def map_pointcloud_to_image(self,
                                pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                cmap: str = 'viridis',
                                cnorm: bool = True) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token,
        load pointcloud and map it to the image plane.

        Arguments:
            pointsensor_token: Lidar/radar sample_data token.
            camera_token: Camera sample_data token.
            min_dist: Distance from the camera below which points are discarded.
            render_intensity: Whether to render lidar intensity instead of point depth.

        Returns:
            (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        if not isinstance(cmap, Colormap):
            cmap = plt.get_c[cmap]

        cam = self.trucksc.get('sample_data', camera_token)
        pointsensor = self.trucksc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.trucksc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.trucksc.dataroot, cam['filename']))

        #limit the elevation of the pointcloud to [-3, 3]m
        pc.points[2,:] = np.clip(pc.points[2,:],-3,3)

        # Points live in the point sensor frame. So they need to be transformed
        # via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame
        # for the timestamp of the sweep.
        cs_record = self.trucksc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.trucksc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame
        # for the timestamp of the image.
        poserecord = self.trucksc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.trucksc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        
        depths = pc.points[2, :]

        if render_intensity:
            if pointsensor['sensor_modality'] == 'lidar':
                # Retrieve the color from the intensities.
                coloring = pc.points[3, :]
            else:
                # Retrieve the color from the rcs.
                coloring = pc.points[6, :]
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Color mapping
        if cnorm:
            norm = Normalize(vmin=np.quantile(coloring, 0.5),
                             vmax=np.quantile(coloring, 0.95), clip=True)
        else:
            norm = None
        mapper = ScalarMappable(norm=norm, cmap=cmap)
        coloring = mapper.to_rgba(coloring)[..., :3]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']),
                             normalize=True)

        # Remove points that are either outside or behind the camera.
        # Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to
        # avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        #coloring = coloring[mask, :]
        coloring = np.tile(np.array([[1.0, 0.0, 0.0]]), (points.shape[1], 1))  # RGB for red

        return points, coloring, im
    








# Initialize the TruckScenes dataset
trucksc = TruckScenes('v1.0-mini', 'D:/Git Repos/truckscenes data/data/man-truckscenes', True)    

trucksc.explorer = MyTruckScenesExplorer(trucksc)  # Use the custom explorer


# Function to project radar point cloud onto camera image and save the result
def project_radar_pcd_to_img(trucksc, radar_name, camera_name, output_folder):
    output_path = Path(output_folder)  # Create a Path object for the output folder
    output_path.mkdir(exist_ok=True)   # Ensure the output folder exists
    for sample in trucksc.sample:
        # Render radar point cloud on the camera image for each sample
        trucksc.render_pointcloud_in_image(
            sample['token'],
            pointsensor_channel=radar_name,
            camera_channel=camera_name,
            render_intensity=True,
            dot_size=2
        )
        # Construct the output filename using sample token, radar, and camera names
        filename = f"sample_{sample['token']}_{radar_name}_{camera_name}.png"
        # Save the plot to the output folder
        plt.savefig(output_path / filename)
        # Close the figure to free memory and avoid warnings
        plt.close()

# Call the function to process and save all images for the specified radar and camera
#project_radar_pcd_to_img(trucksc, radar_name="RADAR_LEFT_FRONT", camera_name="CAMERA_LEFT_FRONT", output_folder="output")
#project_radar_pcd_to_img(trucksc, radar_name="RADAR_RIGHT_FRONT", camera_name="CAMERA_RIGHT_FRONT", output_folder="output2")

my_sample=trucksc.sample[10]
trucksc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='RADAR_LEFT_FRONT', render_intensity=True, dot_size=2)
plt.savefig("radar_point_cloud_custom.png")








