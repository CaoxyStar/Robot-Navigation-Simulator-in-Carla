import random
import queue
import math
import yaml

import numpy as np
import carla
import pygame


class CarlaEnv():
    def __init__(self, cfg):
        # Initialize CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        print('Loading map...')
        map_name = cfg['env']['map_name']
        self.client.load_world(map_name)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        print('Current map: ', self.map.name)
        self.spectator = self.world.get_spectator()
        
        # Set synchronous mode
        self.origin_settings = self.world.get_settings()
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = cfg['env']['step_time']
        self.world.apply_settings(self.settings)

        # Update world
        self.world.tick()

        # Rendering settings
        if cfg['env']['rendering']:
            pygame.init()
            pygame.display.set_caption('Robot View')
            self.camera_num = cfg['robot']['cameras']['rgb'] + cfg['robot']['cameras']['depth'] + cfg['robot']['cameras']['semantic']
            self.screen = pygame.display.set_mode((448 * self.camera_num, 448))
        
        # Load grid map
        map_path = f'Maps/{map_name}/{map_name}_grid_map.npy'
        self.grid_map = np.load(map_path)

        with open('Grid_Map_Construction/map_config.yaml', 'r') as f:
            grid_map_cfg = yaml.safe_load(f)
        self.grid_map_resolution = grid_map_cfg['General']['resolution']
        self.grid_map_x = grid_map_cfg[map_name]['x_range']
        self.grid_map_y = grid_map_cfg[map_name]['y_range']

        # Load sidewalk map for path sampling
        sidewalk_map_path = f'Maps/{map_name}/{map_name}_sidewalk_map.npy'
        self.sidewalk_map = np.load(sidewalk_map_path)
        self.point_list = np.column_stack(np.where(self.sidewalk_map == 255))

        # Initialize actor list
        self.actors = []

        # Set action configuration
        self.move_forward_meter = cfg['robot']['actions']['move_forward']
        self.turn_left_degree = cfg['robot']['actions']['turn_left']
        self.turn_right_degree = cfg['robot']['actions']['turn_right']

        # Save configuration
        self.cfg = cfg
    

    def reset(self):
        # Clear old actors in the world        
        for actor in self.actors:
            actor.destroy()
        self.actors.clear()

        # Update world
        self.world.tick()

        # Sample start and end points for the robot
        start_point_idx = random.randint(0, len(self.point_list) - 1)
        start_point = self.point_list[start_point_idx]

        end_point_idx = random.randint(0, len(self.point_list) - 1)
        end_point = self.point_list[end_point_idx]

        print('Sample path successfully.')
        
        # Initialize spwan point and goal point
        spawn_point = carla.Transform()
        spawn_point.location.x = start_point[0] * self.grid_map_resolution + self.grid_map_x[0]
        spawn_point.location.y = start_point[1] * self.grid_map_resolution + self.grid_map_y[0]
        spawn_point.location.z = 0
        spawn_point.rotation.roll = 0
        spawn_point.rotation.pitch = 0
        spawn_point.rotation.yaw = random.randint(-180, 180)

        waypoint = self.map.get_waypoint(spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        spawn_point.location.z = waypoint.transform.location.z + 1.7

        self.goal_point = carla.Location(
            x=end_point[0] * self.grid_map_resolution + self.grid_map_x[0],
            y=end_point[1] * self.grid_map_resolution + self.grid_map_y[0],
            z=0,
        )

        # Initialize distance
        self.distance = math.sqrt((self.goal_point.x - spawn_point.location.x) ** 2 + (self.goal_point.y - spawn_point.location.y) ** 2)
        self.last_distance = self.distance

        # Spawn the robot
        robot_cameras = self.cfg['robot']['cameras']
        camera_cfg = self.cfg['camera_cfg']
        if robot_cameras['rgb']:
            # Spawn rgb camera
            rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_bp.set_attribute('image_size_x', str(camera_cfg['rgb']['image_size_x']))
            rgb_bp.set_attribute('image_size_y', str(camera_cfg['rgb']['image_size_y']))
            self.rgb_camera = self.world.spawn_actor(rgb_bp, spawn_point)
            self.actors.append(self.rgb_camera)
            self.rgb_queue = queue.Queue(1)
            self.rgb_camera.listen(lambda data: self.rgb_camera_callback(data))
        if robot_cameras['depth']:
            # Spawn depth camera
            depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', str(camera_cfg['depth']['image_size_x']))
            depth_bp.set_attribute('image_size_y', str(camera_cfg['depth']['image_size_y']))
            self.depth_camera = self.world.spawn_actor(depth_bp, carla.Transform(), attach_to=self.rgb_camera)
            self.actors.append(self.depth_camera)
            self.depth_queue = queue.Queue(1)
            self.depth_camera.listen(lambda data: self.depth_camera_callback(data))
        if robot_cameras['semantic']:
            # Spawn semantic camera
            semantic_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            semantic_bp.set_attribute('image_size_x', str(camera_cfg['semantic']['image_size_x']))
            semantic_bp.set_attribute('image_size_y', str(camera_cfg['semantic']['image_size_y']))
            self.semantic_camera = self.world.spawn_actor(semantic_bp, carla.Transform(), attach_to=self.rgb_camera)
            self.actors.append(self.semantic_camera)
            self.semantic_queue = queue.Queue(1)
            self.semantic_camera.listen(lambda data: self.semantic_camera_callback(data))

        robot_sensors = self.cfg['robot']['sensors']
        sensor_cfg = self.cfg['sensor_cfg']
        if robot_sensors['collision']:
            # Spawn collision sensor
            collision_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
            collision_bp.set_attribute('distance', str(sensor_cfg['collision']['distance']))
            collision_bp.set_attribute('hit_radius', str(sensor_cfg['collision']['hit_radius']))
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(carla.Location(z=sensor_cfg['collision']['height_offset'])), attach_to=self.rgb_camera)
            self.actors.append(self.collision_sensor)
            self.collision_sensor.listen(lambda data: self.collision_sensor_callback(data))
        
        # Update world
        self.world.tick()

        # Set spectator
        self.spectator.set_transform(self.rgb_camera.get_transform())

        # Initialize task state
        self.collison = False
        self.out_of_map = False
        self.task_finish = False
        self.terminated = False
        self.truncated = False

        # Get initial observations
        observation = self._get_obs()

        # Initialize iteration state
        self.max_steps = self.cfg['env']['max_steps']
        self.iteration = 0
        
        # Get initial info
        info = self._get_info()
        info['goal'] = observation['goal']

        # Rendering
        if self.cfg['env']['rendering']:
            self.visualize(observation)

        return observation, info
    

    def step(self, action):
        # apply action
        self.apply_action(action)

        # Get observations
        observation = self._get_obs()

        # Update iteration state
        self.iteration += 1
        
        # Update task state
        self._update_state()

        # Compute reward
        reward = self._compute_reward()

        # Get info
        info = self._get_info()
        info['goal'] = observation['goal']
        info['reward'] = reward

        # Rendering
        if self.cfg['env']['rendering']:
            self.visualize(observation)

        return observation, reward, self.terminated, self.truncated, info


    def sample_action(self):
        '''
        There are five actions in the action space: move forward, turn left, turn right, wait and stop.

        Each action is represented by One-hot encoding as follows:

        move forward [1, 0, 0, 0, 0]
        turn left    [0, 1, 0, 0, 0]
        turn right   [0, 0, 1, 0, 0]
        wait         [0, 0, 0, 1, 0]
        stop         [0, 0, 0, 0, 1]
        '''
        index = random.randint(0, 4)
        action = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        action[index] = 1
        return action


    def apply_action(self, action):
        robot_transform = self.rgb_camera.get_transform()

        action = np.argmax(action)
        if action == 0:
            yaw = robot_transform.rotation.yaw
            robot_transform.location.x += self.move_forward_meter * math.cos(yaw * math.pi / 180)
            robot_transform.location.y += self.move_forward_meter * math.sin(yaw * math.pi / 180)
            waypoint = self.map.get_waypoint(robot_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            robot_transform.location.z = waypoint.transform.location.z + 1.7
            self.rgb_camera.set_transform(robot_transform)
        elif action == 1:
            robot_transform.rotation.yaw += self.turn_left_degree
            self.rgb_camera.set_transform(robot_transform)
        elif action == 2:
            robot_transform.rotation.yaw += self.turn_right_degree
            self.rgb_camera.set_transform(robot_transform)
        elif action == 3:
            # Waiting
            pass
        elif action == 4:
            self.task_finish = True
        else:
            raise ValueError("Invalid action index. Please ensure the correct action.")

        self.world.tick()

        # Set spectator
        self.spectator.set_transform(self.rgb_camera.get_transform())

        self.distance = math.sqrt((self.goal_point.x - robot_transform.location.x) ** 2 + (self.goal_point.y - robot_transform.location.y) ** 2)


    def _compute_reward(self):
        ''' Compute the reward based on the task state.
        Returns:
            reward: The computed reward value.
        '''
        # To be implemented
        reward = 0
        
        return reward


    def _get_obs(self):
        ''' Get observations from the environment.
        Returns:
            rgb_image: RGB image from the RGB camera.
            depth_image: Depth image from the depth camera. (If enabled)
            semantic_image: Semantic image from the semantic camera. (If enabled)
        '''
        obs = {}
        # Process rgb image
        if self.cfg['robot']['cameras']['rgb']:   
            rgb_image = self.rgb_queue.get(block=True)
            rgb_image = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
            rgb_image = np.reshape(rgb_image, (224, 224, 4))
            rgb_image = rgb_image[:, :, :3]
            rgb_image = rgb_image[:, :, ::-1]
            obs['rgb'] = rgb_image
        else:
            obs['rgb'] = None

        # Process depth image
        if self.cfg['robot']['cameras']['depth']:
            depth_image = self.depth_queue.get(block=True)
            depth_image = np.frombuffer(depth_image.raw_data, dtype=np.dtype("uint8"))
            depth_image = np.reshape(depth_image, (224, 224, 4))
            depth_image = depth_image[:, :, :3]
            obs['depth'] = depth_image
        else:
            obs['depth'] = None

        # Process semantic image
        if self.cfg['robot']['cameras']['semantic']:
            semantic_image = self.semantic_queue.get(block=True)
            semantic_image = np.frombuffer(semantic_image.raw_data, dtype=np.dtype("uint8"))
            semantic_image = np.reshape(semantic_image, (224, 224, 4))
            semantic_image = semantic_image[:, :, :3]
            semantic_image = semantic_image[:, :, ::-1]
            obs['semantic'] = semantic_image
        else:
            obs['semantic'] = None

        # Compute relative coordinate
        robot_transform = self.rgb_camera.get_transform()
        yaw = np.deg2rad(robot_transform.rotation.yaw)
        dx = self.goal_point.x - robot_transform.location.x
        dy = self.goal_point.y - robot_transform.location.y
        x_robot = dx * np.cos(yaw) + dy * np.sin(yaw)
        y_robot = dx * np.sin(-yaw) + dy * np.cos(yaw)
        obs['goal'] = (x_robot, y_robot)

        return obs
    

    def _get_info(self):
        return {
            'iteration': self.iteration,
            'distance to goal': self.distance,
            'collison': self.collison,
            'out_of_map': self.out_of_map,
            'task_finish': self.task_finish
        }
    
    
    def _update_state(self):
        # Check if the robot is out of the map
        robot_location = self.rgb_camera.get_location()
        x = (robot_location.x - self.grid_map_x[0]) / self.grid_map_resolution
        y = (robot_location.y - self.grid_map_y[0]) / self.grid_map_resolution

        if x < 0 or x >= (self.grid_map_x[1] - self.grid_map_x[0]) / self.grid_map_resolution or \
           y < 0 or y >= (self.grid_map_y[1] - self.grid_map_y[0]) / self.grid_map_resolution or \
           self.grid_map[int(x), int(y)] == 0:
            self.out_of_map = True
        
        if self.out_of_map or self.collison or self.task_finish:
            self.terminated = True
        
        if self.iteration >= self.max_steps:
            self.truncated = True


    def visualize(self, observation):
        image = observation['rgb']
        if observation['depth'] is not None:
            image = np.concatenate((image, observation['depth']), axis=1)
        if observation['semantic'] is not None:
            image = np.concatenate((image, observation['semantic']), axis=1)
        image_surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
        scaled_image = pygame.transform.scale(image_surface, (448 * self.camera_num, 448))
        self.screen.blit(scaled_image, (0, 0))
        pygame.display.flip()
    

    def close(self):
        # Clear old actors in the world        
        for actor in self.actors:
            actor.destroy()
        self.actors.clear()

        # Restore world settings
        self.world.apply_settings(self.origin_settings)
    

    # Callback functions
    def rgb_camera_callback(self, sensor_data):
        self.rgb_queue.put(sensor_data)
    

    def depth_camera_callback(self, sensor_data):
        sensor_data.convert(carla.ColorConverter.Depth)
        self.depth_queue.put(sensor_data)
    

    def semantic_camera_callback(self, sensor_data):
        sensor_data.convert(carla.ColorConverter.CityScapesPalette)
        self.semantic_queue.put(sensor_data)
    

    def collision_sensor_callback(self, sensor_data):
        self.collison = True