import yaml
import argparse
from tqdm import tqdm

import carla
import numpy as np
import matplotlib.pyplot as plt


def get_sidewalk_from_map(map_name):
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world(map_name)

    # Load the map configuration
    with open('Grid_Map_Construction/map_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    general_config = config['General']
    map_config = config[map_name]

    x_min, x_max = map_config['x_range']
    y_min, y_max = map_config['y_range']
    resolution = general_config['resolution']

    # Load the grid map
    grid_map = np.load(f'Maps/{map_name}/{map_name}_grid_map.npy')
    h, w = grid_map.shape

    # Get the world and map in CARLA
    world = client.get_world()
    map = world.get_map()
    
    # Get sidewalk areas
    sidewalk_map = np.zeros((h, w), dtype=np.uint8)
    for i in tqdm(range(0, h * w)):
        x = i // w
        y = i % w
        if grid_map[x, y] == 0:
            continue
        location = carla.Location(x * resolution + x_min, y * resolution + y_min, 0)
        waypoint = map.get_waypoint(location, project_to_road=False, lane_type=carla.LaneType.Sidewalk)
        if waypoint is not None:
            sidewalk_map[x, y] = 255

    # Save the sidewalk map
    np.save(f'Maps/{map_name}/{map_name}_sidewalk_map.npy', sidewalk_map)

    plt.imshow(sidewalk_map, cmap='gray')
    plt.gca().invert_yaxis()
    plt.savefig(f'Maps/{map_name}/{map_name}_sidewalk_map.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Sidewalk from the Grid Map')
    parser.add_argument('--map', type=str, required=True, help='please specify the map name')
    args = parser.parse_args()
    get_sidewalk_from_map(args.map)
    print('Get sidewalk from the grid map for %s successfully!' % args.map)