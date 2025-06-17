export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(rospack find levelManager)/kitchen_models
roslaunch levelManager lego_world.launch
