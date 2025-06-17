#!/usr/bin/env python

import rospy
import rospkg
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose

def spawn_sdf_model():
    rospy.init_node('spawn_model_node')
    
    package_name = "levelManager"
    pkgPath = rospkg.RosPack().get_path(package_name)
    model = "dish_base"
    model_path = f'{pkgPath}/models/{model}/model.sdf'
    
    with open(model_path, 'r') as f:
        model_xml = f.read()

    # delete
    rospy.wait_for_service('/gazebo/delete_model')
    try:
        delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_model_client(model_name=model)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
    rospy.sleep(0.15)
    
    # spawn
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    try:
        spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        resp = spawn_model_prox(model, model_xml, "", None, "_spawn")
        print("Spawn status:\n", resp.status_message)
    except rospy.ServiceException as e:
        rospy.logerr("Spawn service call failed: {0}".format(e))

if __name__ == '__main__':
    spawn_sdf_model()
