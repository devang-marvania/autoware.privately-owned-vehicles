#%%
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from dgp.datasets import SynchronizedSceneDataset
import sys
sys.path.append('../../../../')
from SuperDepth.create_depth.common.lidar_depth_fill import LidarDepthFill
from SuperDepth.create_depth.common.height_map import HeightMap

def cropData(image, depth_map_fill_only, height_map_fill_only, validity_mask):

    # Cropping out those parts of data for which depth is unavailable
    image = image.crop((268, 200, 1668, 900))
    depth_map_fill_only = depth_map_fill_only[200:900, 268:1668]
    height_map_fill_only = height_map_fill_only[200:900, 268:1668]
    validity_mask = validity_mask[200:900, 268:1668]

    return image, depth_map_fill_only, height_map_fill_only, validity_mask

def processSample(sample, max_height, min_height, camera_height):
    
   # Access camera related data
    camera = sample[0][0]
    focal_length = camera['intrinsics'][1][1]
    cy = camera['intrinsics'][1][2]

    # PIL.Image
    image = camera['rgb'] 

    # (H,W) numpy.ndarray, generated from 'lidar' 
    sparse_depth_map = camera['depth']

    # Depth Map
    lidarDepthFill = LidarDepthFill(sparse_depth_map)
    depth_map_fill_only = lidarDepthFill.getDepthMapFillOnly()

    # Height map
    heightMapFillOnly = HeightMap(depth_map_fill_only, max_height, min_height, 
                 camera_height, focal_length, cy)
    height_map_fill_only = heightMapFillOnly.getHeightMap()

    # Validity mask
    validity_mask = np.zeros_like(depth_map_fill_only)
    validity_mask[np.where(depth_map_fill_only != 0)] = 1
    
    # Crop side regions where depth data is missing
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
      cropData(image, depth_map_fill_only, height_map_fill_only, validity_mask)
    
    return image, depth_map_fill_only, height_map_fill_only, validity_mask 

def saveData(root_save_path, counter, image, depth_map_fill_only, 
            height_map_fill_only, validity_mask):
  
  # Save files
  # RGB Image as PNG
  image_save_path = root_save_path + '/image/' + str(counter) + '.png'
  image.save(image_save_path, "PNG")

  # Depth map as binary file in .npy format
  depth_save_path = root_save_path + '/depth/' + str(counter) + '.npy'
  np.save(depth_save_path, depth_map_fill_only)

  # Height map as binary file in .npy format
  height_save_path = root_save_path + '/height/' + str(counter) + '.npy'
  np.save(height_save_path, height_map_fill_only)

  # Validity mask as black and white PNG
  validity_save_path = root_save_path + '/validity/' + str(counter) + '.png'
  validity_mask = Image.fromarray(np.uint8(validity_mask*255))
  validity_mask.save(validity_save_path, "PNG")

  # Height map plot for data auditing purposes
  height_plot_save_path = root_save_path + '/height_plot/' + str(counter) + '.png'
  plt.imsave(height_plot_save_path, height_map_fill_only, cmap='inferno_r')

def main():
  
  # Argument parser for data root path and save path
  parser = ArgumentParser()
  parser.add_argument("-d", "--data", dest="ddad_json_path", help="path to ddad.json file")
  parser.add_argument("-s", "--save", dest="root_save_path", help="path to folder where processed data will be saved")
  args = parser.parse_args()

  # Path to dataset main json file
  ddad_json_path = args.ddad_json_path

  # Save path
  root_save_path = args.root_save_path

  # Load synchronized pairs of camera and lidar frames.
  dataset_train_front = \
  SynchronizedSceneDataset(ddad_json_path,
      datum_names=('lidar', 'CAMERA_01'),
      generate_depth_from_datum='lidar',
      split='train'
      )
  
  dataset_val_front = \
  SynchronizedSceneDataset(ddad_json_path,
      datum_names=('lidar', 'CAMERA_01'),
      generate_depth_from_datum='lidar',
      split='val'
      )
  
  dataset_train_rear = \
  SynchronizedSceneDataset(ddad_json_path,
      datum_names=('lidar', 'CAMERA_09'),
      generate_depth_from_datum='lidar',
      split='train'
      )
  
  dataset_val_rear = \
  SynchronizedSceneDataset(ddad_json_path,
      datum_names=('lidar', 'CAMERA_09'),
      generate_depth_from_datum='lidar',
      split='val'
      )

  # Iterate through the dataset.
  num_train_front_samples = len(dataset_train_front)
  num_val_front_samples = len(dataset_val_front)
  num_train_rear_samples = len(dataset_train_rear)
  num_val_rear_samples = len(dataset_val_rear)
  total_data = num_train_front_samples + num_val_front_samples \
    + num_train_rear_samples + num_val_rear_samples

  print('Total samples', total_data)

  # Data counter
  counter = 0

  # Camera height above road surface
  camera_height_front = 1.3
  camera_height_rear = 1.4

  # Height map limits
  max_height = 7
  min_height = -2
  
  # Training dataset
  for index in range(0, num_train_rear_samples, 2):

    # Get data sample and process it
    sample = dataset_train_rear[index]
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
        processSample(sample, max_height, min_height, camera_height_rear)

    # Save data
    saveData(root_save_path, counter, image, depth_map_fill_only, 
            height_map_fill_only, validity_mask)

    print('Processing image ', counter, ' of ', total_data/2)

    counter += 1
  
  for index in range(0, num_train_front_samples, 2):

    # Get data sample and process it
    sample = dataset_train_front[index]
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
        processSample(sample, max_height, min_height, camera_height_front)

    # Save data
    saveData(root_save_path, counter, image, depth_map_fill_only, 
            height_map_fill_only, validity_mask)

    print('Processing image ', counter, ' of ', total_data/2)

    counter += 1

    # Training dataset
  for index in range(0, num_val_rear_samples, 2):

    # Get data sample and process it
    sample = dataset_val_rear[index]
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
        processSample(sample, max_height, min_height, camera_height_rear)

    # Save data
    saveData(root_save_path, counter, image, depth_map_fill_only, 
            height_map_fill_only, validity_mask)

    print('Processing image ', counter, ' of ', total_data/2)

    counter += 1
  
  # Validation dataset
  for index in range(0, num_val_front_samples, 2):

    # Get data sample and process it
    sample = dataset_val_front[index]
    image, depth_map_fill_only, height_map_fill_only, validity_mask = \
        processSample(sample, max_height, min_height, camera_height_front)
    
    # Save data
    saveData(root_save_path, counter, image, depth_map_fill_only, 
        height_map_fill_only, validity_mask)

    print('Processing image ', counter, ' of ', total_data/2)

    counter += 1
  
if __name__ == '__main__':
  main()

#%%