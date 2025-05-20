import torch

class Network(torch.nn.Module):
    def __init__(self, systems, controllers):
        """
        systems: list of RobotsSystem instances (one per robot)
        controllers: list of controller instances (one per robot)
        """
        assert len(systems) == len(controllers), "Each robot must have a controller."
        self.systems = systems
        self.controllers = controllers
        self.n_agents = len(systems)

    def rollout(self, data_list, device):
        """
        data_list: list of (robot_id, batch, T, state_dim+ref_dim) tensors, one per robot
        device: torch.device
        Returns: lists of x_log, u_log, v_log for each robot
        """
        n_agents = self.n_agents
        batch_size = data_list[0].shape[0]

        # Initialize states, integrals, controls for all robots
        x = [self.systems[i].x_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)]
        v = [torch.zeros(batch_size, 1, 2, device=device) for _ in range(n_agents)]
        u = [self.systems[i].u_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)]


        # Logs
        x_log = [[] for _ in range(n_agents)]
        u_log = [[] for _ in range(n_agents)]
        v_log = [[] for _ in range(n_agents)]

        T = data_list[0].shape[1]  # number of time steps
        for t in range(T):
            # Gather all agent positions for communication
            positions = [x[i][..., :2] for i in range(n_agents)]  # (batch, 1, 2) for each agent

            # Compute control for each agent
            for i in range(n_agents):
                # Reference for agent i at this step
                xbar_i = data_list[i][:, t:t+1, 4:] # (batch, 1, 4)
                w = data_list[i][:, t:t+1, :4]  # Disturbance for agent i (batch, 1, 4)

                # Pass all other agents' positions (excluding self)
                neighbor_positions = [positions[j] for j in range(n_agents) if j != i]
                # For 2 robots, neighbor_positions[0] is the other robot's position
                # For more, you may want to concatenate or otherwise process
                # Here, we concatenate all neighbor positions along last dim
                if neighbor_positions:
                    neighbor_pos = torch.cat(neighbor_positions, dim=-1)
                else:
                    neighbor_pos = torch.zeros_like(positions[i])
                # Controller computes its control
                u[i] = self.controllers[i](x[i], v[i], xbar=xbar_i, neighbor_pos=neighbor_pos)

                x[i], v[i] = self.systems[i].forward(t, x[i], v[i], u[i], w=w[i], xbar=data_list[i][:, t:t+1, 4:], neighbor_pos=neighbor_pos)
                x_log[i].append(x[i])
                u_log[i].append(u[i])
                v_log[i].append(v[i])

        # Stack logs: (batch, T, state_dim)
        x_log = [torch.cat(x_log[i], dim=1) for i in range(n_agents)]
        u_log = [torch.cat(u_log[i], dim=1) for i in range(n_agents)]
        v_log = [torch.cat(v_log[i], dim=1) for i in range(n_agents)]
        return x_log, u_log, v_log