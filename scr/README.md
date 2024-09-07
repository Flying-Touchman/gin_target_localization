# Nodes 

`GIN` package include the following ROS nodes.

## target-loc

Node responsible for detection of target and computation of center pose

### Subscribers

All subscribers name are paramtetrized via ROS paramaters:

- Camera color image <color_topic_name> ( [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) )
- Camera depth image  <depth_topic_name> ( [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) )
- Camera parameters <info_topic_name> ( [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) )


### Publishers

- Positive detection of target (Y/n) `/target/detection` ([std_msgs/Bool](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Bool.html));
- Target center position and orientation `/target/pose` ( [geometry_msgs/PoseStamped](http://docs.ros.org/kinetic/api/geometry_msgs/html/msg/PoseStamped.html) )
- Image with detection box layer `/target/image_with_detection`( [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) )
  
### Parameters

- `param_file_name` Neural Network weights file
- `config_file_name`: Neural Network config file
- `names_file_name`: Path to the file containing the class names for the target detection.
- `depth_topic_name`: Depth image topic name
- `info_topic_name`: Camera parameters topic name 
- `color_topic_name`: Color image topic name
- `default_camera_param_file`: Default camera parameters json file(in case `info_topic_name` not present)
- `box_param_file`: Dimensions of detection boxes json file
- `method`: Method used to compute the target pose (pnp or camera_proj).
- `pub_rate`: Rate (in Hz) at which target pose and detection information is published.
- `pub_rate_image`: Rate (in Hz) at which the detection image is published.
- `detection_method`: The detection strategy used (single_layer or multiple_layer).

## target-distance-depth

Node responsible for computing distance to the target.
Useful to check whether drone is too close to the target or not

### Subscribers

- Depth image `/d400/aligned_depth_to_color/image_raw` ( [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) )

### Publishers

- Cropped depth image used for distance computation (debug) `/target/cropped_depth_image` ( [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html) )
- Distance to the target `/target/depth` ( [std_msgs/Float32](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/Float32.html) )