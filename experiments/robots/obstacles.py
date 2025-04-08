import torch
from config import device
import numpy as np

def generate_rhombic_lattice(min_distance, x1, x2):
    """
    Generates a rhombic lattice pattern within a given bounding box.
    
    Parameters:
    min_distance (float): The minimum distance between points.
    x1 (tuple): Bottom-left coordinate of the bounding box (x_min, y_min).
    x2 (tuple): Top-right coordinate of the bounding box (x_max, y_max).
    
    Returns:
    torch.Tensor: Tensor of points in the rhombic lattice within the bounding box.
    """
    # Define basis vectors for the rhombic structure
    a1 = np.array([min_distance, 0])  # First basis vector
    a2 = np.array([min_distance / 2, min_distance * np.sqrt(3) / 2])  # Second basis vector

    # Bounding box limits
    x_min, y_min = x1
    x_max, y_max = x2

    # Estimate the number of points needed
    max_i = int((x_max - x_min) / min_distance) + 2
    max_j = int((y_max - y_min) / (min_distance * np.sqrt(3) / 2)) + 2

    # Generate the lattice points
    lattice_points = []
    for i in range(-max_i, max_i):
        for j in range(-max_j, max_j):
            point = i * a1 + j * a2
            # Check if point is inside the bounding box
            if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max:
                lattice_points.append(point)
    # Convert to tensor
    lattice_points = np.array(lattice_points)

    return torch.tensor(lattice_points, device=device)

def generate_obstacles(interval_x1=(-3, 3), interval_x2=(-3, 3), min_distance=1.0, num_obstacles=10, spacing='random'):
    """
    Generate obstacles spaced either randomly or evenly with a minimum distance.

    Args:
        interval_x1 (tuple): Interval for the x1 coordinate.
        interval_x2 (tuple): Interval for the x2 coordinate.
        min_distance (float): Minimum distance between obstacles.
        num_obstacles (int): Number of obstacles to generate.
        spacing (str): 'random' for random spacing, 'even' for even spacing.

    Returns:
        torch.Tensor: Tensor containing the obstacle coordinates.
    """
    obstacles = []

    if spacing == 'random':
        while len(obstacles) < num_obstacles:
            # Generate a random obstacle
            obstacle = torch.tensor([
                torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
            ]).flatten()

            # Check the distance to all existing obstacles
            if all(torch.norm(obstacle - obs) >= min_distance for obs in obstacles):
                obstacles.append(obstacle)
    elif spacing == 'even':
        x1_values = torch.linspace(interval_x1[0], interval_x1[1], int(torch.sqrt(torch.tensor(num_obstacles)).item()))
        x2_values = torch.linspace(interval_x2[0], interval_x2[1], int(torch.sqrt(torch.tensor(num_obstacles)).item()))
        for x1 in x1_values:
            for x2 in x2_values:
                obstacle = torch.tensor([x1, x2])
                if all(torch.norm(obstacle - obs) >= min_distance for obs in obstacles):
                    obstacles.append(obstacle)
                if len(obstacles) >= num_obstacles:
                    break
            if len(obstacles) >= num_obstacles:
                break
    elif spacing == 'lattice':
        x1_values = torch.linspace(interval_x1[0], interval_x1[1], int(torch.sqrt(torch.tensor(num_obstacles)).item()))
        x2_values = torch.linspace(interval_x2[0], interval_x2[1], int(torch.sqrt(torch.tensor(num_obstacles)).item()))
        for i, x1 in enumerate(x1_values):
            for j, x2 in enumerate(x2_values):
                if (i + j) % 2 == 0:  # Checkerboard pattern
                    obstacle = torch.tensor([x1, x2])
                    if all(torch.norm(obstacle - obs) >= min_distance for obs in obstacles):
                        obstacles.append(obstacle)
                if len(obstacles) >= num_obstacles:
                    break
            if len(obstacles) >= num_obstacles:
                break
    elif spacing == 'rhombic':
        bottom_left = (interval_x1[0], interval_x2[0])
        top_right = (interval_x1[1], interval_x2[1])
        lattice_points = generate_rhombic_lattice(min_distance, bottom_left, top_right)
        obstacles = [point for point in lattice_points]

    return torch.stack(obstacles)


obstacles = {
    0: {
        "centers": [
            torch.tensor([[-0.5, 0]], device=device),
            torch.tensor([[0.5, 0.0]], device=device),
        ],
        "covs": [
            torch.tensor([[0.1, 0.1]], device=device)
        ]
    },
    1: {
        "centers": [
            torch.tensor([[0.5, 2]], device=device),
            torch.tensor([[1, 2.0]], device=device),
            torch.tensor([[3, 2]], device=device),
            torch.tensor([[3.5, 2.0]], device=device),
        ],
        "covs": [
            torch.tensor([[0.1, 0.1]], device=device)
        ]
    },
    2: {
        "centers": [
            torch.tensor([[-3, -1]], device=device),
            torch.tensor([[-2, -0.6]], device=device),
            torch.tensor([[-1, 0]], device=device),
            torch.tensor([[-2, 0.6]], device=device),
            torch.tensor([[-3, 1]], device=device),
            torch.tensor([[3, -1]], device=device),
            torch.tensor([[2, -0.6]], device=device),
            torch.tensor([[1, 0]], device=device),
            torch.tensor([[2, 0.6]], device=device),
            torch.tensor([[3, 1]], device=device),
        ],
        "covs": [
            torch.tensor([[0.1, 0.1]], device=device)
        ]
    },
    3: {
        "centers": [
            torch.tensor([[-3.0, -1.0]], device=device),
            torch.tensor([[-2.5, -1.0]], device=device),
            torch.tensor([[-2.0, -1.0]], device=device),
            torch.tensor([[-1.5, -1.0]], device=device),
            torch.tensor([[-1.0, -1.0]], device=device),
            torch.tensor([[-3.0, 1.0]], device=device),
            torch.tensor([[-2.5, 1.0]], device=device),
            torch.tensor([[-2.0, 1.0]], device=device),
            torch.tensor([[-1.5, 1.0]], device=device),
            torch.tensor([[-1.0, 1.0]], device=device),
            torch.tensor([[-3.0, -0.5]], device=device),
            torch.tensor([[-3.0, 0.0]], device=device),
            torch.tensor([[-3.0, 0.5]], device=device),
            torch.tensor([[-1.0, -0.5]], device=device),
            torch.tensor([[-1.0, 0.0]], device=device),
            torch.tensor([[-1.0, 0.5]], device=device),
            torch.tensor([[1.0, -1.0]], device=device),
            torch.tensor([[1.5, -1.0]], device=device),
            torch.tensor([[2.0, -1.0]], device=device),
            torch.tensor([[2.5, -1.0]], device=device),
            torch.tensor([[3.0, -1.0]], device=device),
            torch.tensor([[1.0, 1.0]], device=device),
            torch.tensor([[1.5, 1.0]], device=device),
            torch.tensor([[2.0, 1.0]], device=device),
            torch.tensor([[2.5, 1.0]], device=device),
            torch.tensor([[3.0, 1.0]], device=device),
            torch.tensor([[1.0, -0.5]], device=device),
            torch.tensor([[1.0, 0.0]], device=device),
            torch.tensor([[1.0, 0.5]], device=device),
            torch.tensor([[3.0, -0.5]], device=device),
            torch.tensor([[3.0, 0.0]], device=device),
            torch.tensor([[3.0, 0.5]], device=device),
        ],
        "covs": [
            torch.tensor([[0.05, 0.05]], device=device)
        ]
    },
    4: {
        "centers": generate_obstacles(interval_x1=(-2, 2), interval_x2=(-1, 1), min_distance=1.0, num_obstacles=5, spacing='random'),
        "covs": [
            torch.tensor([[0.1, 0.1]], device=device)
        ]
    },
    5: {
        "centers": generate_obstacles(interval_x1=(-3, 3), interval_x2=(-1.5, 1.5), min_distance=1, num_obstacles=5, spacing='rhombic'),
        "covs": [
            torch.tensor([[0.05, 0.05]], device=device)
        ]
    },
    6: {
        "centers": [
            torch.tensor([[-3.0, 0]], device=device),
            torch.tensor([[-1.6, 0]], device=device),
            torch.tensor([[-0.2, 0]], device=device),
            torch.tensor([[1.2, 0]], device=device),
            torch.tensor([[2.6, 0]], device=device),
            torch.tensor([[4.0, 0]], device=device),
        ],
        "covs": [
            torch.tensor([[0.05, 0.05]], device=device)
        ]
    },
    7: {
    "centers": [
        torch.tensor([[-3.0, 2]], device=device),
        torch.tensor([[-1, 2]], device=device),
        torch.tensor([[1, 2]], device=device),
        torch.tensor([[3, 2]], device=device),
        torch.tensor([[5, 2]], device=device),
        torch.tensor([[7.0, 2]], device=device),
    ],
    "covs": [
        torch.tensor([[0.1, 0.1]], device=device)
    ]
    }
}


