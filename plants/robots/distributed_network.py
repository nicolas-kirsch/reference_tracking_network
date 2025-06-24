import torch
from config import device

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


    # def rollout(self, controller, data, train=False):
    #     """
    #     rollout REN for rollouts of the process noise for distributed control.

    #     Args:
    #         - data: list of (batch, T, state_dim+ref_dim = 8) tensors, one per robot
    #             (batch_size, T, state_dim).

    #     Return:
    #         - x_log of shape = (batch_size, T, state_dim* n_agents)
    #         - e_log of shape = (batch_size, T, in_dim* n_agents)
    #         - u_log of shape = (batch_size, T, in_dim* n_agents)
            
    #     """

    #     # init

    #     controller.reset()


    #     n_agents = self.n_agents
    #     batch_size = data[0].shape[0]
    #     T = data[0].shape[1]
    #     state_dim = self.systems[0].x_init.shape[-1]
    #     control_dim = self.systems[0].u_init.shape[-1]

    #     # Initialize states, integrals, controls for all robots
    #     x = torch.stack([self.systems[i].x_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)], dim=1) # shape = (batch_size, n_agents, 1, state_dim)
    #     u = torch.stack([self.systems[i].u_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)], dim=1) # shape = (batch_size, n_agents, 1, in_dim)
    #     v = torch.zeros(u.shape) # shape = (batch_size, n_agents, 1, in_dim)
    #     w = torch.stack([data[i][:,:,:4] for i in range(n_agents)], dim=1) # shape = (batch_size, n_agents, T, 4)
    #     xbar = torch.stack([data[i][:,:,4:] for i in range(n_agents)], dim=1) # shape = (batch_size, n_agents, T, 4)


    #     v = v.to(device) # Antoine added this line
    #     x = x.to(device) # Antoine added this line
    #     w = w.to(device) # Antoine added this line

    #     # Simulate
    #     for t in range(T):
    #         for i in range(n_agents):
    #         x,v = self.forward(t=t, x=x, u=u, v=v, w=w[:, t:t+1, :],xbar= xbar[:, t:t+1, :])    # shape = (batch_size, 1, state_dim)

    #         #u_k = c(x_k,xbar_k)
    #         u = controller(x,v,xbar[:, t:t+1, :])                                       # shape = (batch_size, 1, in_dim)

    #         xbar = xbar.to(device) # Antoine added this line
    #         x = x.to(device) # Antoine added this line
    #         if t == 0:
    #             x_log, u_log, v_log = x, u,v
    #             e_log = xbar[:, t:t+1, :2] - x[:,:,[0, 1]]


    #         else:
    #             x_log = torch.cat((x_log, x), 1)
    #             u_log = torch.cat((u_log, u), 1)
    #             v_log = torch.cat((v_log, v), 1)
    #             e_log = torch.cat((e_log, xbar[:, t:t+1, :2] - x[:,:,[0, 1]]), 1)

    #     controller.reset()
    #     if not train:
    #         x_log, u_log = x_log.detach(), u_log.detach()

    #     self.v_log = v_log.detach()
    #     return x_log, e_log, u_log

    def rollout(self, data_list, device, train=False):
        """
        data_list: list of (batch, T, state_dim+ref_dim) tensors, one per robot
        device: torch.device
        Returns: x_log, u_log, v_log as tensors
        """
        self.reset()  # Reset controllers before rollout
        
        n_agents = self.n_agents
        batch_size = data_list[0].shape[0]
        T = data_list[0].shape[1]
        state_dim = self.systems[0].x_init.shape[-1]
        control_dim = self.systems[0].u_init.shape[-1]

        # Initialize states, integrals, controls for all robots
        x_init = torch.stack([self.systems[i].x_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)], dim=1)
        # x_init: (batch, n_agents, 1, state_dim)
        u_init = torch.stack([self.systems[i].u_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)], dim=1)
        # u_init: (batch, n_agents, 1, control_dim)
        v_init = torch.zeros(batch_size, n_agents, 1, control_dim, device=device)

        x = x_init.clone()
        u = u_init.clone()
        v = v_init.clone()

        # Logs
        x_log = []
        u_log = []
        v_log = []
        e_log = []
        # with torch.no_grad():
        for t in range(T):
            # Positions: (batch, n_agents, 1, 2)
            positions = x[..., :2]  # (batch, n_agents, 1, 2)
            velocities = x[..., 2:]  # (batch, n_agents, 1, 2)
            # Compute control for each agent
            u_step = torch.zeros(batch_size, n_agents, 1, control_dim, device=device)
            for i in range(n_agents):
                # Reference for agent i at this step
                xbar_i = data_list[i][:, t:t+1, 4:].to(device)  # (batch, 1, ref_dim)
                w_i = data_list[i][:, t:t+1, :4].clone().to(device)  # Create a copy
                # print(f"Loop 1, Agent {i}: w_i = {w_i}")  # Print w_i

                # Neighbor positions: (batch, n_agents-1, 1, 2)
                neighbor_pos = torch.cat([positions[:, j] for j in range(n_agents) if j != i], dim=1)  # (batch, n_agents-1, 1, 2)
                neighbor_vel = torch.cat([velocities[:, j] for j in range(n_agents) if j != i], dim=1)  # (batch, n_agents-1, 1, 2)
                # Controller computes its control
                # Get the other agent's control
                if i == 0:
                    u_neighbor = u_step[:, 1:2].clone()  # Get agent 1's control
                else:  # i == 1
                    u_neighbor = u_step[:, 0:1].clone()  # Get agent 0's control
                u_neighbor = u_neighbor.squeeze(2)  # Remove the time dimension, shape: (batch, 1, control_dim)
                u_step[:, i] = self.controllers[i](
                    x[:, i], v[:, i], xbar=xbar_i, neighbor_pos=neighbor_pos, neighbor_vel=neighbor_vel, u_neighbor = u_neighbor
                )

            # Step each system
            x_next = torch.zeros_like(x)
            v_next = torch.zeros_like(v)
            e_next = torch.zeros_like(v)
            for i in range(n_agents):
                xbar_i = data_list[i][:, t:t+1, 4:].to(device)  # (1, 1, ref_dim = 4 (x, y, vx, vy))
                w_i = data_list[i][:, t:t+1, :4].clone().to(device)  # Create a copy
                # print(f"Loop 2, Agent {i}: w_i = {w_i}")  # Print w_i

                neighbor_pos = torch.cat([positions[:, j] for j in range(n_agents) if j != i], dim=1) # (batch, n_agents-1, 1, 2)
                x_next[:, i], v_next[:, i] = self.systems[i].forward(
                    t, x[:, i], v[:, i], u_step[:, i], w=w_i, xbar=xbar_i, neighbor_pos=neighbor_pos
                ) # Shape x_next: (1, n_agents, 1, state_dim = 4), v_next: (1, n_agents, 1, control_dim = 2)
                e_next[:, i] = xbar_i[:,:,:2] - x[:, i, :,:2]

            x = x_next
            v = v_next
            u = u_step
            e = e_next

            x_log.append(x)
            u_log.append(u)
            v_log.append(v)
            e_log.append(e)
            assert not torch.isnan(x).any(), f"NaN in x at step {t}"
            assert not torch.isnan(u).any(), f"NaN in u at step {t}"
            assert not torch.isnan(v).any(), f"NaN in v at step {t}"    
        self.reset()  # Reset controllers after rollout

        if not train:
            # If not training, detach logs to avoid unnecessary gradients
            x_log = [x.detach() for x in x_log]
            u_log = [u.detach() for u in u_log]
            v_log = [v.detach() for v in v_log]
            e_log = [e.detach() for e in e_log]

        # Stack along agent dimension: (batch, n_agents, T, 1, state_dim)
        x_log = torch.stack(x_log, dim=2)
        u_log = torch.stack(u_log, dim=2)
        v_log = torch.stack(v_log, dim=2)
        e_log = torch.stack(e_log, dim=2)

        x_log = x_log.squeeze(3)  # shape: (batch, n_agents, T, state_dim)
        u_log = u_log.squeeze(3)  # shape: (batch, n_agents, T, control_dim)
        v_log = v_log.squeeze(3)  # shape: (batch, n_agents, T, control_dim)
        e_log = e_log.squeeze(3)  # shape: (batch, n_agents, T, control_dim)

        x_log = x_log.permute(0, 2, 1, 3)  # (batch, T, n_agents, state_dim)
        x_log = x_log.reshape(x_log.shape[0], x_log.shape[1], -1)  # (batch, T, n_agents * state_dim)
        u_log = u_log.permute(0, 2, 1, 3)  # (batch, T, n_agents, control_dim)
        u_log = u_log.reshape(u_log.shape[0], u_log.shape[1], -1)  # (batch, T, n_agents * control_dim)
        v_log = v_log.permute(0, 2, 1, 3)  # (batch, T, n_agents, control_dim)
        v_log = v_log.reshape(v_log.shape[0], v_log.shape[1], -1)  # (batch, T, n_agents * control_dim)
        e_log = e_log.permute(0, 2, 1, 3)  # (batch, T, n_agents, control_dim)
        e_log = e_log.reshape(e_log.shape[0], e_log.shape[1], -1)  # (batch, T, n_agents * control_dim)
        return x_log, u_log, v_log, e_log

    def reset(self):
        """
        Reset the state of each system and controller.
        """
        for i in range(self.n_agents):
            self.controllers[i].reset()

    # def rollout(self, data_list, device):
    #     """
    #     data_list: list of (robot_id, batch, T, state_dim+ref_dim) tensors, one per robot
    #     device: torch.device
    #     Returns: lists of x_log, u_log, v_log for each robot
    #     """
    #     n_agents = self.n_agents
    #     batch_size = data_list[0].shape[0]

    #     # Initialize states, integrals, controls for all robots
    #     x = [self.systems[i].x_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)]
    #     v = [torch.zeros(batch_size, 1, 2, device=device) for _ in range(n_agents)]
    #     u = [self.systems[i].u_init.detach().clone().repeat(batch_size, 1, 1).to(device) for i in range(n_agents)]


    #     # Logs
    #     x_log = [[] for _ in range(n_agents)]
    #     u_log = [[] for _ in range(n_agents)]
    #     v_log = [[] for _ in range(n_agents)]

    #     T = data_list[0].shape[1]  # number of time steps
    #     for t in range(T):
    #         # Gather all agent positions for communication
    #         positions = [x[i][..., :2] for i in range(n_agents)]  # (batch, 1, 2) for each agent

    #         # Compute control for each agent
    #         for i in range(n_agents):
    #             # Reference for agent i at this step
    #             xbar_i = data_list[i][:, t:t+1, 4:] # (batch, 1, 4)
    #             w = data_list[i][:, t:t+1, :4]  # Disturbance for agent i (batch, 1, 4)

    #             # Pass all other agents' positions (excluding self)
    #             neighbor_positions = [positions[j] for j in range(n_agents) if j != i]
    #             # For 2 robots, neighbor_positions[0] is the other robot's position
    #             # For more, you may want to concatenate or otherwise process
    #             # Here, we concatenate all neighbor positions along last dim
    #             if neighbor_positions:
    #                 neighbor_pos = torch.cat(neighbor_positions, dim=-1)
    #             else:
    #                 neighbor_pos = torch.zeros_like(positions[i])
    #             # Controller computes its control
    #             u[i] = self.controllers[i](x[i], v[i], xbar=xbar_i, neighbor_pos=neighbor_pos)

    #             x[i], v[i] = self.systems[i].forward(t, x[i], v[i], u[i], w=w[i], xbar=data_list[i][:, t:t+1, 4:], neighbor_pos=neighbor_pos)
    #             x_log[i].append(x[i])
    #             u_log[i].append(u[i])
    #             v_log[i].append(v[i])

    #     # Stack logs: (batch, T, state_dim)
    #     x_log = [torch.cat(x_log[i], dim=1) for i in range(n_agents)]
    #     u_log = [torch.cat(u_log[i], dim=1) for i in range(n_agents)]
    #     v_log = [torch.cat(v_log[i], dim=1) for i in range(n_agents)]
    #     return x_log, u_log, v_log