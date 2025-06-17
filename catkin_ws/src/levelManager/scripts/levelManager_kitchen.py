#!/usr/bin/python3
from pandas import array
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import *
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler
import rospy, rospkg, rosservice
import sys
import time
import random
import numpy as np
import os
import xml.etree.ElementTree as ET

path = rospkg.RosPack().get_path("levelManager")

costruzioni = ['costruzione-1', 'costruzione-2']

def randomCostruzione():
	return random.choice(costruzioni)

def getPose(modelEl):
	strpose = modelEl.find('pose').text
	return [float(x) for x in strpose.split(" ")]

def get_Name_Type(modelEl):
	if modelEl.tag == 'model':
		name = modelEl.attrib['name']
	else:
		name = modelEl.find('name').text
	return name, name.split('_')[0]
	

def get_Parent_Child(jointEl):
	parent = jointEl.find('parent').text.split('::')[0]
	child = jointEl.find('child').text.split('::')[0]
	return parent, child


def getLego4Costruzione(select=None):
	nome_cost = randomCostruzione()
	if select is not None: nome_cost = costruzioni[select]
	print("spawning", nome_cost)

	tree = ET.parse(f'{path}/lego_models/{nome_cost}/model.sdf')
	root = tree.getroot()
	costruzioneEl = root.find('model')

	brickEls = []
	for modEl in costruzioneEl:
		if modEl.tag in ['model', 'include']:
			brickEls.append(modEl)

	models = ModelStates()
	for bEl in brickEls:
		pose = getPose(bEl)
		models.name.append(get_Name_Type(bEl)[1])
		rot = Quaternion(*quaternion_from_euler(*pose[3:]))
		models.pose.append(Pose(Point(*pose[:3]), rot))

	rospy.init_node("levelManager")
	istruzioni = rospy.Publisher("costruzioneIstruzioni", ModelStates, queue_size=1)
	istruzioni.publish(models)

	return models


def changeModelColor(model_xml, color):
	root = ET.XML(model_xml)
	root.find('.//material/script/name').text = color
	return ET.tostring(root, encoding='unicode')	


#DEFAULT PARAMETERS
package_name = "levelManager"
spawn_name = '_spawn'
level = None
selectBrick = None
maxLego = 11
spawn_pos = (-0.35, -0.42, 0.74)  		#center of spawn area
spawn_dim = (0.32, 0.23)    			#spawning area
min_space = 0.010    					#min space between lego
min_distance = 0.15   					#min distance between lego

#function parsing arguments
def readArgs():
	global package_name
	global level
	global selectBrick
	try:
		argn = 1
		while argn < len(sys.argv):
			arg = sys.argv[argn]
			
			if arg.startswith('__'):
				None
			elif arg[0] == '-':
				if arg in ['-l', '-level']:
					argn += 1
					level = int(sys.argv[argn])
				elif arg in ['-b', '-brick']:
					argn += 1
					selectBrick = brickList[int(sys.argv[argn])]
				else:
					raise Exception()
			else: raise Exception()
			argn += 1
	except Exception as err:
		print("Usage: .\levelManager.py" \
					+ "\n\t -l | -level: assigment from 1 to 4" \
					+ "\n\t -b | -brick: spawn specific grip from 0 to 10")
		exit()
		pass

"""
brickDict:
key: name of the object
value: (type, (x,y,height) - size information and height)
"""
brickDict = { \
    'plate_1': (0, (0.15,0.15,0.2)),
    'plate_2': (1, (0.15,0.15,0.45))
}

"""
brickOrientations:
key: name of the object
value: (((side, roll), ...), rotX, height)
"""
brickOrientations = {
	'plate_1': (((1,1),(1,3)),-1.715224,0.18), \
	'plate_2': (((1,1),(1,3)),2.496793,0.20)
}

# color bricks
colorList = ['Gazebo/Indigo', 'Gazebo/Gray', 'Gazebo/Orange', \
		'Gazebo/Red', 'Gazebo/Purple', 'Gazebo/SkyBlue', \
		'Gazebo/DarkYellow', 'Gazebo/White', 'Gazebo/Green']

brickList = list(brickDict.keys())
counters = [0 for brick in brickList]

lego = [] 	#lego = [[name, type, pose, radius], ...]

#get model path
def getModelPath(model):
    pkgPath = rospkg.RosPack().get_path(package_name)
    lego_path = f'{pkgPath}/lego_models/{model}/model.sdf'
    kitchen_path = f'{pkgPath}/kitchen_models/{model}/model.sdf'
    if os.path.exists(lego_path):
        return lego_path
    elif os.path.exists(kitchen_path):
        return kitchen_path
    else:
        raise Exception("Model not found:" + model)

#set position brick
def randomPose(brickType, rotated):
	_, dim, = brickDict[brickType]
	spawnX = spawn_dim[0]
	spawnY = spawn_dim[1]
	rotX = 0
	rotY = 0
	rotZ = random.uniform(-3.14, 3.14)
	pointX = random.uniform(-spawnX, spawnX)
	pointY = random.uniform(-spawnY, spawnY)
	pointZ = dim[2]/2
	dim1 = dim[0]
	dim2 = dim[1]
	if rotated:
		side = random.randint(0, 1) 	#0=z/x, 1=z/y
		if (brickType == "X2-Y2-Z2"):
			roll = random.randint(1, 1)	#0=z, 1=x/y, 2=z. 3=x/y
		else:
			roll = random.randint(2, 2) #0=z, 1=x/y, 2=z. 3=x/y

		orients = brickOrientations.get(brickType, ((),0,0) )		
		if (side, roll) not in orients[0]:
			rotX = (side)*roll*1.57
			rotY = (1-side)*roll*1.57
			if roll % 2 != 0:
				dim1 = dim[2]
				dim2 = dim[1-side]
				pointZ = dim[side]/2
		else:
			rotX = orients[1]
			pointZ = orients[2]
			
	rot = Quaternion(*quaternion_from_euler(rotX, rotY, rotZ))
	point = Point(pointX, pointY, pointZ)
	return Pose(point, rot), dim1, dim2

class PoseError(Exception):
	pass

"""
function to get a valid pose
check if there is a brick in the area to avoid collision
"""
def getValidPose(brickType, rotated):
	trys = 1000
	valid = False
	while not valid:
		pos, dim1, dim2 = randomPose(brickType, rotated)
		radius = np.sqrt((dim1**2 + dim2**2)) / 2
		valid = True
		# check if there is a brick in the area
		for brick in lego:
			point = brick[2].position
			r2 = brick[3]
			minDist = max(radius + r2 + min_space, min_distance)
			if (point.x-pos.position.x)**2+(point.y-pos.position.y)**2 < minDist**2:
				valid = False
				trys -= 1
				if trys == 0:
					raise PoseError(f"Fail to spawn {brickType} due to collision")
				break
	return pos, radius

#functiont to spawn model
def spawn_model(model, pos, name=None, ref_frame='world', color=None):
	if name is None:
		name = model
	model_xml = open(getModelPath(model), 'r').read()
	if color is not None:
		model_xml = changeModelColor(model_xml, color)
	print(f"==== spawning model: {model}\npose: {pos}\nname: {name}")
 
	spawn_model_client = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
	return spawn_model_client(model_name=name, 
	    model_xml=model_xml,
	    robot_namespace='/foo',
	    initial_pose=pos,
	    reference_frame=ref_frame)

#support function delete bricks on table
def delete_model(name):
	delete_model_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
	return delete_model_client(model_name=name)

#support functon spawn bricks
def spawnObject(brickType=None, rotated=False):
	if brickType is None:
		brickType = random.choice(brickList)
	
	brickIndex = brickDict[brickType][0]
	target_counter = counters
        
	name = f'{brickType}_{target_counter[brickIndex]+1}'
	pos, radius = getValidPose(brickType, rotated)
	
	color = random.choice(colorList)
	# color = None

	spawn_model(brickType, pos, name, spawn_name, color)
	lego.append((name, brickType, pos, radius))
	target_counter[brickIndex] += 1

#main function setup area and level manager
def setUpArea(livello=None, selectBrick=None): 	

	#delete all bricks on the table
	for brickType in brickList:	#ripulisce
		count = 1
		while delete_model(f'{brickType}_{count}').success: 
			print("deleting", brickType, count)
			count += 1
	
	#screating spawn area
	spawn_model(spawn_name, Pose(Point(*spawn_pos),None) )
 
	try:
		if(livello == 1):
			#spawn random brick
			spawnObject(selectBrick)
			#spawnObject('X2-Y2-Z2',rotated=True)
		elif(livello == 2):
			#spawn all bricks
			for brickType in brickList:
				spawnObject(brickType)
		elif(livello == 3):
			#spawn first 4 blocks	
			for brickType in brickList[0:4]:
				spawnObject(brickType)
			#spawn three blocks rotated
			spawnObject('X1-Y2-Z2',rotated=True)
			spawnObject('X1-Y2-Z2',rotated=True)
			spawnObject('X2-Y2-Z2',rotated=True)
		elif(livello == 4):
			if selectBrick is None:
				#spawn blocks build
				spawn_dim = (0.10, 0.10)    			#spawning area
				import time
				spawnObject('plate_2')
				time.sleep(2)
				spawnObject('plate_2')
				time.sleep(2)
				spawnObject('plate_2')	
				time.sleep(2)
				spawnObject('plate_2')
				time.sleep(2)
				spawnObject('plate_2')	
				time.sleep(2)
				spawnObject('plate_2')
			else:
				models = getLego4Costruzione()
				r = 3
				for brickType in models.name:
					r -= 1
					spawnObject(brickType, rotated=r>0)
		else:
			print("[Error]: select level from 1 to 4")
			return
	except PoseError as err:
		print("[Error]: no space in spawning area")
		pass
		
	print(f"Added {len(lego)} bricks")

if __name__ == '__main__':

	readArgs()

	try:
		if '/gazebo/spawn_sdf_model' not in rosservice.get_service_list():
			print("Waining gazebo service..")
			rospy.wait_for_service('/gazebo/spawn_sdf_model')

		setUpArea(level, selectBrick)
	except rosservice.ROSServiceIOException as err:
		print("No ROS master execution")
		pass
	except rospy.ROSInterruptException as err:
		print(err)
		pass
	except rospy.service.ServiceException:
		print("No Gazebo services in execution")
		pass
	except rospkg.common.ResourceNotFound:
		print(f"Package not found: '{package_name}'")
		pass
	except FileNotFoundError as err:
		print(f"Model not found: \n{err}")
		print(f"Check model in folder'{package_name}/lego_models'")
		pass
