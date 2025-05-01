import sys, os, logging, torch, time, json
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from utils.plot_functions import *


args = argument_parser()
# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'robots', 'Figures')
os.makedirs(save_path, exist_ok=True)

plot_obstacles_and_random_points(obstacle_centers=args.obstacle_centers,obstacle_covs = args.obstacle_covs, x_range=args.x_interval, y_range=(4,6), n_points= 15, save_folder=save_path, filename=f"random_points_{now}.png") 