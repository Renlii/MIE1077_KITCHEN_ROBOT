#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import rospy
import json
import os
import time
import random
import sys
from datetime import datetime

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gazebo_msgs.srv import SpawnModel, DeleteModel, GetModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import rospkg
import message_filters

# Import existing functions from levelManager_kitchen
sys.path.append(os.path.dirname(__file__))
from levelManager_kitchen import (
    objDict, spawn_model, delete_model, getModelPath, 
    spawn_pos, spawn_dim, min_space, min_distance
)

class KitchenDatasetGenerator:
    def __init__(self, output_dir="kitchen_dataset", num_iterations=1000):
        self.output_dir = output_dir
        self.num_iterations = num_iterations
        self.bridge = CvBridge()
        self.spawn_name = '_spawn'
        self.camera_frame = 'camera_link'
        
        # Camera parameters (from kitchen-vision.py)
        self.cam_point = (-0.44, -0.5, 1.58)
        self.height_tavolo = 0.74
        
        # Object list from kitchen models
        self.obj_list = list(objDict.keys())
        
        # Initialize ROS services
        rospy.init_node('kitchen_dataset_generator', anonymous=True)
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        rospy.wait_for_service('/gazebo/get_model_state')
        
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        # Create output directories
        self.setup_directories()
        
        # COCO format structure
        self.coco_data = {
            "info": {
                "description": "Kitchen Objects Dataset for YOLO Training",
                "version": "1.0",
                "year": 2024,
                "contributor": "UR5 Pick and Place Simulation",
                "date_created": datetime.now().isoformat()
            },
            "categories": [
                {"id": i, "name": name, "supercategory": "kitchen_object"} 
                for i, name in enumerate(self.obj_list)
            ],
            "images": [],
            "annotations": []
        }
        
        self.annotation_id = 0
        self.image_id = 0
        
    def setup_directories(self):
        """Create necessary directories for dataset"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "annotations"), exist_ok=True)
        
    def load_object_mesh_data(self, obj_name):
        """Load bounding box data from JSON mesh files"""
        package_path = rospkg.RosPack().get_path("levelManager")
        mesh_dir = os.path.join(package_path, "kitchen_models", obj_name, "meshes")
        
        # Find the main JSON file for the object
        json_files = [f for f in os.listdir(mesh_dir) if f.endswith('.json')]
        
        # Try to find the main object JSON file
        main_json = None
        for json_file in json_files:
            if obj_name in json_file or json_file.startswith(obj_name.split('_')[0]):
                main_json = json_file
                break
        
        if not main_json and json_files:
            # Set the main json with the biggest size
            sizes_sum = [sum(json.load(open(f))["model_size"]) for f in json_files]
            main_json = json_files[np.argmax(sizes_sum)]
        
        if main_json:
            json_path = os.path.join(mesh_dir, main_json)
            try:
                with open(json_path, 'r') as f:
                    mesh_data = json.load(f)
                return mesh_data
            except Exception as e:
                print(f"Error loading mesh data for {obj_name}: {e}")

        # Fallback to objDict dimensions
        print("Fall back to read objDict from levelManager")
        _, (x, y, z) = objDict[obj_name]
        return {
            "model_size": [x, y, z],
            "bounding_box_min": [-x/2, -y/2, 0],
            "bounding_box_max": [x/2, y/2, z]
        }
    
    def random_pose_with_rotation(self, obj_type):
        """Generate random pose with horizontal rotation"""
        _, dim = objDict[obj_type]
        spawnX = spawn_dim[0]
        spawnY = spawn_dim[1]
        
        # Random horizontal rotation (around Z-axis)
        rotZ = random.uniform(-np.pi, np.pi)
        
        # Random position within spawn area
        pointX = random.uniform(-spawnX, spawnX)
        pointY = random.uniform(-spawnY, spawnY)
        pointZ = dim[2]/2  # Half height above ground
        
        rot = Quaternion(*quaternion_from_euler(0, 0, rotZ))
        point = Point(pointX, pointY, pointZ)
        
        return Pose(point, rot)
    
    def check_collision(self, new_pose, new_obj_type, existing_objects):
        """Check if new object would collide with existing ones"""
        _, new_dim = objDict[new_obj_type]
        new_radius = np.sqrt((new_dim[0]**2 + new_dim[1]**2)) / 2
        
        for obj_name, obj_type, obj_pose in existing_objects:
            _, obj_dim = objDict[obj_type]
            obj_radius = np.sqrt((obj_dim[0]**2 + obj_dim[1]**2)) / 2
            
            # Calculate distance between centers
            dx = new_pose.position.x - obj_pose.position.x
            dy = new_pose.position.y - obj_pose.position.y
            distance = np.sqrt(dx**2 + dy**2)
            
            min_dist = max(new_radius + obj_radius + min_space, min_distance)
            if distance < min_dist:
                return True
        
        return False
    
    def spawn_random_objects(self, max_objects=5):
        """Spawn random objects in the scene"""
        # Clear existing objects
        self.clear_all_objects()
        
        # Create spawn area
        spawn_model(self.spawn_name, Pose(Point(*spawn_pos), None))
        
        # Track spawned objects
        spawned_objects = []
        
        # Randomly choose number of objects (1 to max_objects)
        num_objects = random.randint(1, max_objects)
        
        for i in range(num_objects):
            attempts = 0
            max_attempts = 100
            
            while attempts < max_attempts:
                # Randomly select object type
                obj_type = random.choice(self.obj_list)
                
                # Generate random pose
                pose = self.random_pose_with_rotation(obj_type)
                
                # Check for collisions
                if not self.check_collision(pose, obj_type, spawned_objects):
                    # Spawn object
                    obj_name = f"{obj_type}_{i+1}"
                    
                    try:
                        result = spawn_model(obj_type, pose, obj_name, self.spawn_name)
                        if result.success:
                            spawned_objects.append((obj_name, obj_type, pose))
                            print(f"Spawned {obj_name} at {pose.position}")
                            break
                    except Exception as e:
                        print(f"Error spawning {obj_name}: {e}")
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Could not spawn object {i+1} after {max_attempts} attempts")
        
        return spawned_objects
    
    def clear_all_objects(self):
        """Clear all objects from the scene"""
        for obj_type in self.obj_list:
            count = 1
            while True:
                obj_name = f'{obj_type}_{count}'
                result = delete_model(obj_name)
                if not result.success:
                    break
                count += 1
    
    def get_object_poses_from_gazebo(self, spawned_objects):
        """Get actual poses of objects from Gazebo after physics simulation"""
        actual_poses = []
        
        for obj_name, obj_type, _ in spawned_objects:
            try:
                state = self.get_model_state(obj_name, 'world')
                if state.success:
                    actual_poses.append((obj_name, obj_type, state.pose))
                else:
                    print(f"Could not get state for {obj_name}")
            except Exception as e:
                print(f"Error getting state for {obj_name}: {e}")
        
        return actual_poses
    
    def calculate_bounding_box_2d(self, pose, mesh_data):
        """Calculate 2D bounding box in image coordinates"""
        # Get 3D bounding box from mesh data
        bbox_min = np.array(mesh_data["bounding_box_min"])
        bbox_max = np.array(mesh_data["bounding_box_max"])
        
        # Extract pose information
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_quat(orientation)
        
        # Get 8 corners of 3D bounding box
        corners_3d = []
        for x in [bbox_min[0], bbox_max[0]]:
            for y in [bbox_min[1], bbox_max[1]]:
                for z in [bbox_min[2], bbox_max[2]]:
                    corner = np.array([x, y, z])
                    # Apply rotation and translation
                    corner_world = rotation.apply(corner) + position
                    corners_3d.append(corner_world)
        
        corners_3d = np.array(corners_3d)
        
        # Project 3D points to 2D image coordinates
        # Camera parameters (simplified pinhole model)
        # These should match the actual camera parameters in Gazebo
        fx = fy = 554.254691191187  # Focal length
        cx, cy = 320, 240  # Principal point
        
        corners_2d = []
        for corner in corners_3d:
            # Transform to camera coordinate system
            # Camera position relative to world
            cam_x, cam_y, cam_z = self.cam_point
            
            # Relative position to camera
            rel_x = corner[0] - cam_x
            rel_y = corner[1] - cam_y
            rel_z = corner[2] - cam_z
            
            # Project to image plane
            if rel_z > 0:  # In front of camera
                u = fx * (-rel_x) / rel_z + cx
                v = fy * (-rel_y) / rel_z + cy
                corners_2d.append([u, v])
        
        if not corners_2d:
            return None
        
        corners_2d = np.array(corners_2d)
        
        # Get bounding box
        x_min = max(0, int(np.min(corners_2d[:, 0])))
        y_min = max(0, int(np.min(corners_2d[:, 1])))
        x_max = min(640, int(np.max(corners_2d[:, 0])))
        y_max = min(480, int(np.max(corners_2d[:, 1])))
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width <= 0 or height <= 0:
            return None
            
        return [x_min, y_min, width, height]
    
    def capture_images_and_annotations(self, spawned_objects):
        """Capture RGB and depth images and create annotations"""
        # Wait for images
        print("Waiting for camera images...")
        
        try:
            # Get RGB image
            rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=10)
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            
            # Get depth image
            depth_msg = rospy.wait_for_message("/camera/depth/image_raw", Image, timeout=10)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            
        except rospy.ROSException as e:
            print(f"Error capturing images: {e}")
            return None, None, []
        
        # Get actual poses after physics simulation
        actual_poses = self.get_object_poses_from_gazebo(spawned_objects)
        
        # Create annotations
        annotations = []
        for obj_name, obj_type, pose in actual_poses:
            # Load mesh data
            mesh_data = self.load_object_mesh_data(obj_type)
            
            # Calculate 2D bounding box
            bbox_2d = self.calculate_bounding_box_2d(pose, mesh_data)
            
            if bbox_2d is not None:
                # Get category ID
                category_id = self.obj_list.index(obj_type)
                
                # Create COCO annotation
                annotation = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": category_id,
                    "bbox": bbox_2d,
                    "area": bbox_2d[2] * bbox_2d[3],
                    "iscrowd": 0,
                    "segmentation": []  # Empty for bounding box only
                }
                annotations.append(annotation)
                self.annotation_id += 1
        
        return rgb_image, depth_image, annotations
    
    def save_data(self, rgb_image, depth_image, annotations, iteration):
        """Save images and update COCO annotations"""
        # Save RGB image
        rgb_filename = f"kitchen_{iteration:06d}.jpg"
        rgb_path = os.path.join(self.output_dir, "images", rgb_filename)
        cv.imwrite(rgb_path, rgb_image)
        
        # Save depth image
        depth_filename = f"kitchen_{iteration:06d}_depth.png"
        depth_path = os.path.join(self.output_dir, "depth", depth_filename)
        # Normalize depth for saving
        depth_norm = cv.normalize(depth_image, None, 0, 65535, cv.NORM_MINMAX, dtype=cv.CV_16U)
        cv.imwrite(depth_path, depth_norm)
        
        # Add image info to COCO data
        image_info = {
            "id": self.image_id,
            "file_name": rgb_filename,
            "width": rgb_image.shape[1],
            "height": rgb_image.shape[0],
            "depth_file_name": depth_filename
        }
        self.coco_data["images"].append(image_info)
        
        # Add annotations
        self.coco_data["annotations"].extend(annotations)
        
        print(f"Saved iteration {iteration}: {len(annotations)} objects detected")
        self.image_id += 1
    
    def generate_dataset(self):
        """Main function to generate the dataset"""
        print(f"Starting dataset generation: {self.num_iterations} iterations")
        print(f"Output directory: {self.output_dir}")
        
        successful_iterations = 0
        
        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            
            try:
                # Spawn random objects
                spawned_objects = self.spawn_random_objects()
                
                if not spawned_objects:
                    print("No objects spawned, skipping iteration")
                    continue
                
                # Wait for objects to settle
                print("Waiting for objects to settle...")
                time.sleep(1.0)
                
                # Capture images and create annotations
                rgb_image, depth_image, annotations = self.capture_images_and_annotations(spawned_objects)
                
                if rgb_image is not None and annotations:
                    # Save data
                    self.save_data(rgb_image, depth_image, annotations, iteration)
                    successful_iterations += 1
                else:
                    print("Failed to capture valid data, skipping iteration")
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                continue
        
        # Save final COCO annotations
        coco_path = os.path.join(self.output_dir, "annotations", "instances.json")
        with open(coco_path, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Successful iterations: {successful_iterations}/{self.num_iterations}")
        print(f"Total images: {len(self.coco_data['images'])}")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        print(f"COCO annotations saved to: {coco_path}")
        
        # Clean up
        self.clear_all_objects()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate kitchen objects dataset for YOLO training')
    parser.add_argument('-n', '--num_iterations', type=int, default=1000,
                       help='Number of iterations to generate (default: 1000)')
    parser.add_argument('-o', '--output_dir', type=str, default='kitchen_dataset',
                       help='Output directory for dataset (default: kitchen_dataset)')
    parser.add_argument('--max_objects', type=int, default=5,
                       help='Maximum number of objects per scene (default: 5)')
    
    args = parser.parse_args()
    
    try:
        generator = KitchenDatasetGenerator(
            output_dir=args.output_dir,
            num_iterations=args.num_iterations
        )
        generator.generate_dataset()
        
    except rospy.ROSInterruptException:
        print("Dataset generation interrupted")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()