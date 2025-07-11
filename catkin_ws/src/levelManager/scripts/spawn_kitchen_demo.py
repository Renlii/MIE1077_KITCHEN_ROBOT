#!/usr/bin/env python
import sys
import os
import random
import rospy
import rospkg
from gazebo_msgs.srv import SpawnModel, DeleteModel
from levelManager_kitchen import spawn_pos, spawn_dim
from geometry_msgs.msg import *
from tf.transformations import quaternion_from_euler
from pyquaternion import Quaternion

package_name = "levelManager"
pose = []

def randomPose():
	spawnX = spawn_dim[0]
	spawnY = spawn_dim[1]
	pointX = random.uniform(-spawnX, spawnX)
	pointY = random.uniform(-spawnY, spawnY)
	pointZ = 0.1
	point = Point(pointX, pointY, pointZ)
	return Pose(point, None)
def find_model_path(model_name):
    pkgPath = rospkg.RosPack().get_path(package_name)
    search_paths = [f'{pkgPath}/models/{model_name}/model.sdf',
                    f'{pkgPath}/kitchen_models/{model_name}/model.sdf',
                    f'{pkgPath}/lego_models/{model_name}/model.sdf']
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None

def spawn_sdf_model():
    rospy.init_node('spawn_model_node')
    
    if len(sys.argv) >= 2:
        model = sys.argv[1]
    else:
        model = "cup_green"
        
    model_path = find_model_path(model)
    print(model_path)
    with open(model_path, 'r') as f:
        model_xml = f.read()

    # delete
    print("deleting...")
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model_client(model_name=model)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    import time
    time.sleep(1)
    
    # pose 
    pose = randomPose()
    # ori = Quaternion(*quaternion_from_euler(0.01, 0, 0))
    pose = Pose(Point(0.50, -0.90, 0.78), None)
    print(pose)
    
    # spawn
    print("spawning...")
    ref = "_spawn"
    ref = "world"
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp = spawn_model_prox(model, model_xml, "", pose, ref)
        print("Spawn status:\n", resp.status_message)
    except rospy.ServiceException as e:
        rospy.logerr("Spawn service call failed: {0}".format(e))

if __name__ == '__main__':
    spawn_sdf_model()
