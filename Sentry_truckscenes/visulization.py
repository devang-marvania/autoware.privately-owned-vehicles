## Exploring the dataset

#initialization

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg') 
from truckscenes import TruckScenes



trucksc = TruckScenes('v1.0-mini', 'data/man-truckscenes', True)

#scene
trucksc.list_scenes()

my_scene = trucksc.scene[0]
print(my_scene)

#sample
first_sample_token = my_scene['first_sample_token']
trucksc.render_sample(first_sample_token)
plt.savefig('first_scene_first_sample.png')

my_sample=trucksc.get('sample',first_sample_token)
print(my_sample)

trucksc.list_sample(my_sample['token'])


#sample_data
print(my_sample['data'])

sensor='CAMERA_LEFT_FRONT'
cam_front_data=trucksc.get('sample_data',my_sample['data'][sensor])
print(cam_front_data)
trucksc.render_sample_data(cam_front_data['token'])
plt.savefig('cam_front_data_render.png')

#sample_annotation

my_annotation_token=my_sample['anns'][14]
my_annotation_metadata = trucksc.get('sample_annotation',my_annotation_token)
print(my_annotation_metadata)
trucksc.render_annotation(my_annotation_token)
plt.savefig('my_annotation_token.png')

#instance

my_instance=trucksc.get('instance',my_annotation_metadata['instance_token'])
print(my_instance)

instance_token=my_instance['token']
trucksc.render_instance(instance_token)
plt.savefig('instance_token.png')
trucksc.render_annotation(my_instance['first_annotation_token'])
plt.savefig('instance_first_annotation_token')

trucksc.render_annotation(my_instance['last_annotation_token'])
plt.savefig('instance_last_annotation_token')


#category
trucksc.list_categories()
print(trucksc.category[9])


#attribute
trucksc.list_attributes()

#example to show how an attribute may change over one scene
my_instance = trucksc.instance[3]
first_token = my_instance['first_annotation_token']
last_token = my_instance['last_annotation_token']
nbr_samples = my_instance['nbr_annotations']
current_token = first_token

i = 0
while current_token != last_token:
    current_ann = trucksc.get('sample_annotation', current_token)
    current_attr = trucksc.get('attribute', current_ann['attribute_tokens'][0])['name']
    
    if i == 0:
        pass
    elif current_attr != last_attr:
        print("Changed from `{}` to `{}` at timestamp {} out of {} annotated timestamps".format(last_attr, current_attr, i, nbr_samples))

    next_token = current_ann['next']
    current_token = next_token
    last_attr = current_attr
    i += 1


#visibility
print(trucksc.visibility)

visibility_token=trucksc.get('sample_annotation',my_annotation_token)['visibility_token']

print("Visibility : {}".format(trucksc.get('visibility',visibility_token)))
trucksc.render_annotation(my_annotation_token)
plt.savefig("visibility_token_example.png")

#visibility example for 0-40% visibility
anntoken = '03287f5e06c94b5f9ffa61e937ba7c2d'
visibility_token = trucksc.get('sample_annotation', anntoken)['visibility_token']

print("Visibility: {}".format(trucksc.get('visibility', visibility_token)))
trucksc.render_annotation(anntoken, box_vis_level=0)
plt.savefig("low_visibility_token_example.png")

#sensor
print(trucksc.sensor)
print(trucksc.sample_data[10])

#calibrated sensor
print(trucksc.calibrated_sensor[0])

#ego_pose
print(trucksc.ego_pose[0])

#ego_motion
print(trucksc.ego_motion_cabin[0])
print(trucksc.ego_motion_chassis[0])


## Basic Usage
print(trucksc.category[0])
cat_token=trucksc.category[0]['token']
print(cat_token)
print(trucksc.get('category',cat_token))

print(trucksc.sample_annotation[0])
print(trucksc.get('visibility',trucksc.sample_annotation[0]['visibility_token']))

one_instance=trucksc.get('instance',trucksc.sample_annotation[0]['instance_token'])
print(one_instance)

ann_tokens=trucksc.field2token('sample_annotation','instance_token',one_instance['token'])
ann_tokens_field2token=set(ann_tokens)
print(ann_tokens_field2token)

ann_record=trucksc.get('sample_annotation',one_instance['first_annotation_token'])
print(ann_record)

ann_tokens_traverse = set()
ann_tokens_traverse.add(ann_record['token'])
while not ann_record['next'] == "":
    ann_record = trucksc.get('sample_annotation', ann_record['next'])
    ann_tokens_traverse.add(ann_record['token'])

print(ann_tokens_traverse == ann_tokens_field2token)    


sensor = 'CAMERA_LEFT_FRONT'
cam_left_front_data = trucksc.get('sample_data', my_sample['data'][sensor])

closest_ego_pose = trucksc.getclosest('ego_pose', cam_left_front_data['timestamp'])

# Difference in microseconds
delta_t = closest_ego_pose['timestamp'] - cam_left_front_data['timestamp']
print(f"Time difference: {delta_t / 1e6}s")


##Reverse Indexing and Shortcuts
#Category name example

# Using shortcut
catname = trucksc.sample_annotation[0]['category_name']

# Not using shortcut
ann_rec = trucksc.sample_annotation[0]
inst_rec = trucksc.get('instance', ann_rec['instance_token'])
cat_rec = trucksc.get('category', inst_rec['category_token'])

print(catname == cat_rec['name'])

#channel and modality example

# Using shortcut
channel = trucksc.sample_data[0]['channel']

# Not using shortcut
sd_rec = trucksc.sample_data[0]
cs_record = trucksc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
sensor_record = trucksc.get('sensor', cs_record['sensor_token'])

print(channel == sensor_record['channel'])

##Data Visualization

print(trucksc.list_categories())

print(trucksc.list_attributes())

print(trucksc.list_scenes())

my_sample=trucksc.sample[10]
trucksc.render_pointcloud_in_image(my_sample['token'],pointsensor_channel='LIDAR_LEFT',camera_channel='CAMERA_LEFT_FRONT',dot_size=2)
plt.savefig("pointcloud_im_image.png")

trucksc.render_pointcloud_in_image(my_sample['token'],pointsensor_channel='LIDAR_LEFT',render_intensity=True,dot_size=2)
plt.savefig("lidar_intensity.png")

trucksc.render_pointcloud_in_image(my_sample['token'],pointsensor_channel='RADAR_LEFT_FRONT',render_intensity=True,dot_size=2)
plt.savefig("radar_point_cloud.png")

my_sample=trucksc.sample[20]
trucksc.render_sample(my_sample['token'])
plt.savefig("render_all_sensors.png")

trucksc.render_sample_data(my_sample['data']['CAMERA_RIGHT_FRONT'])
plt.savefig("render_one_sensor.png")

trucksc.render_sample_data(my_sample['data']['LIDAR_RIGHT'],nsweeps=5)
plt.savefig("render_one_LIDAR_sensor_multiple_sweeps.png")

trucksc.render_sample_data(my_sample['data']['RADAR_RIGHT_FRONT'], nsweeps=5)
plt.savefig("render_one_RADAR_sensor_multiple_sweeps.png")


trucksc.render_annotation(my_sample['anns'][0])
plt.savefig("specific_annotation.png")

#Ran into runtime issues!
#trucksc.render_calibrated_sensor(first_sample_token) 


#3D image does render but still gets some warnings
#trucksc.render_pointcloud(my_sample, chans=['LIDAR_LEFT', 'LIDAR_RIGHT'], ref_chan='LIDAR_LEFT', with_anns=True)

my_scene_token = trucksc.field2token('scene', 'name', 'scene-3f542f89ec5241b6a4e30ca743adcf34-29')[0]

#video rendering works but need to figure out how to save it
#trucksc.render_scene_channel(my_scene_token, 'CAMERA_LEFT_FRONT')

#video rendering works but need to figure out how to save it 
#trucksc.render_scene(my_scene_token)
