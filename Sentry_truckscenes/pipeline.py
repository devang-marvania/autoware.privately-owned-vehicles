
# Import necessary libraries
import matplotlib.pyplot as plt  # For plotting and saving images
from truckscenes import TruckScenes  # For loading and handling truck scenes dataset
from pathlib import Path  # For handling file and folder paths

# Initialize the TruckScenes dataset
trucksc = TruckScenes('v1.0-mini', 'data/man-truckscenes', True)

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








