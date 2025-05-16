import torch
from plants import CostumDataset


class RobotsDataset(CostumDataset):
    def __init__(self, random_seed, horizon,x_bar,x0, std_ini=0.2, n_agents=2):
        # experiment and file names
        exp_name = 'robots'
        file_name = 'data_T'+str(horizon)+'_stdini'+str(std_ini)+'_agents'+str(n_agents)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.std_ini = std_ini
        self.n_agents = n_agents

        # initial state TODO: set as arg
        # self.x0 = torch.tensor([-2, -2, 0, 0,
        #                         2, -2, 0, 0,
        #                         ])
        self.x0 = x0
        self.xbar = x_bar
        

    def generate_vector_with_min_distance(self,interval_x1=(-3, 3), interval_x2=(2, 3), min_distance=1.0):

        # vec1 = torch.tensor([
        #         torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
        #         torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
        #     ]).flatten()
        
        # return vec1
        while True:
            # Generate the first vector (x1 in [-2, 2], x2 in [0, 1])
            vec1 = torch.tensor([
                torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
            ]).flatten()

            # Generate the second vector (x1 in [-2, 2], x2 in [0, 1])
            vec2 = torch.tensor([
                torch.empty(1).uniform_(interval_x1[0], interval_x1[1]),
                torch.empty(1).uniform_(interval_x2[0], interval_x2[1])
            ]).flatten()
            # Check the Euclidean distance
            distance = torch.norm(vec1 - vec2)
            if distance >= min_distance:
                # Concatenate vec1 and vec2 to form a (4,) vector
                return torch.cat((vec1, vec2))

    # ---- data generation ----
    def _generate_data(self, num_samples, x_interval=(-3, 7), y_interval=(4, 4.1)):

        #Initial conditions
        state_dim_x0 = 4*self.n_agents

        #Reference tracking data
        state_dim_ref = 4*self.n_agents

        #Total dimensions
        state_dim = state_dim_x0+state_dim_ref

        data = torch.zeros(num_samples, self.horizon, state_dim)

        """for rollout_num in range(num_samples):
            data[rollout_num, 0, :state_dim_x0] = \
                    self.x0 + self.std_ini * torch.randn(self.x0.shape)
            data[rollout_num, 1:, state_dim_x0:] = \
                self.xbar"""


        """for rollout_num in range(num_samples):
            data[rollout_num, 0, :state_dim_x0] = \
                self.x0 + self.std_ini * torch.randn(self.x0.shape)
            data[rollout_num, 1:, state_dim_x0:state_dim_x0+1] = \
                    self.xbar[0:1] + torch.empty(1).uniform_(-0.3, 4)
                #+ torch.empty(1).uniform_(-1, 4)
            data[rollout_num, 1:, state_dim_x0+1:state_dim_x0+2] = \
                    self.xbar[1:2]+ torch.empty(1).uniform_(0, 0.3)
                #+ torch.empty(1).uniform_(-1.5, 1.5)
            data[rollout_num, 1:, state_dim_x0+4:state_dim_x0+5] = \
                    self.xbar[4:5] + torch.empty(1).uniform_(-4, 0.3)
                #+ torch.empty(1).uniform_(-4, 1)
            data[rollout_num, 1:, state_dim_x0+5:state_dim_x0+6] = \
                    self.xbar[5:6] + torch.empty(1).uniform_(0, 0.3)"""
            
        for rollout_num in range(num_samples):
            # vecs = self.generate_vector_with_min_distance(interval_x1=(-2, 2), interval_x2=(2, 2.1), min_distance=2.0) # Antoine: I changed the interval_x2 from (2, 2.1) to (-2, 2)
            vecs = self.generate_vector_with_min_distance(interval_x1=x_interval, interval_x2=y_interval, min_distance=2.0)
            # if rollout_num % 2 == 0:
            #     # First zone of training
            #     vecs = self.generate_vector_with_min_distance(interval_x1=(-2, 2), interval_x2=(2, 2.1), min_distance=2.0)
            # else:
            #     # Second zone of training
            #     vecs = self.generate_vector_with_min_distance(interval_x1=(-2, 2), interval_x2=(-2, -1.9), min_distance=2.0)
            data[rollout_num, 0, :state_dim_x0] = \
                self.x0 + self.std_ini * torch.randn(self.x0.shape)
            data[rollout_num, 1:, state_dim_x0:state_dim_x0+2] = \
                vecs[0:2]
            data[rollout_num, 1:, state_dim_x0+4:state_dim_x0+6] = \
                vecs[2:4]

        """for rollout_num in range(num_samples):
            data[rollout_num, 0, :state_dim_x0] = \
                    self.x0 + self.std_ini * torch.randn(self.x0.shape)
            
            if rollout_num%2 == 0:
                data[rollout_num, 1:, state_dim_x0:state_dim_x0+1] = \
                    self.xbar[0:1] + torch.empty(1).uniform_(-0.3, 0.3)
                #+ torch.empty(1).uniform_(-1, 4)
                data[rollout_num, 1:, state_dim_x0+1:state_dim_x0+2] = \
                    self.xbar[1:2]+ torch.empty(1).uniform_(0, 0.3)
                #+ torch.empty(1).uniform_(-1.5, 1.5)
                data[rollout_num, 1:, state_dim_x0+4:state_dim_x0+5] = \
                    self.xbar[4:5] + torch.empty(1).uniform_(-0.3, 0.3)
                #+ torch.empty(1).uniform_(-4, 1)
                data[rollout_num, 1:, state_dim_x0+5:state_dim_x0+6] = \
                    self.xbar[5:6] + torch.empty(1).uniform_(0, 0.3)
            else:
                data[rollout_num, 1:, state_dim_x0:state_dim_x0+1] = \
                    self.xbar[4:5] + torch.empty(1).uniform_(-0.3, 0.3)
                #+ torch.empty(1).uniform_(-1, 4)
                data[rollout_num, 1:, state_dim_x0+1:state_dim_x0+2] = \
                    self.xbar[5:6] + torch.empty(1).uniform_(0, 0.3)
                #+ torch.empty(1).uniform_(-1.5, 1.5)
                data[rollout_num, 1:, state_dim_x0+4:state_dim_x0+5] = \
                    self.xbar[0:1] + torch.empty(1).uniform_(-0.3, 0.3)
                #+ torch.empty(1).uniform_(-4, 1)
                data[rollout_num, 1:, state_dim_x0+5:state_dim_x0+6] = \
                    self.xbar[1:2] + torch.empty(1).uniform_(0, 0.3)"""
            
        assert data.shape[0]==num_samples
        return data