import numpy as np

from typing import List, Tuple, Dict

import torch
from torch.nn import LazyLinear
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, Linear
import torch.nn.functional as F   
import torch.optim as optim

from utils import trunc_normal

from IPython.display import clear_output
import matplotlib.pyplot as plt

class PolicyNetRGCN(torch.nn.Module):
    class Actor(torch.nn.Module): 
        def __init__(self, config: dict):
            super().__init__()    
            self.train_device = config['train_device']  
            self.action_dim = config['action_dim']
            
            graph = config['graph']
            self.num_node_features = graph['num_node_features']
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.edge_type = torch.tensor(graph['edge_type'], dtype=torch.long)
            self.num_nodes = graph['num_nodes']
            self.num_relations = graph['num_relations']           

            self.in_channels = self.num_node_features              
            self.conv1 = RGCNConv(self.in_channels, 32, self.num_relations, num_bases=None, aggr='mean')
            self.conv2 = RGCNConv(32, 32, self.num_relations, num_bases=None, aggr='mean')
            self.conv3 = RGCNConv(32, 16, self.num_relations, num_bases=None, aggr='mean')
            self.conv4 = RGCNConv(16, 16, self.num_relations, num_bases=None, aggr='mean')

            self.mu_head = torch.nn.Sequential(
                torch.nn.Linear(16*self.num_nodes, 128),
                torch.nn.GELU(),
                torch.nn.LayerNorm(128),
                torch.nn.Linear(128, self.action_dim)
            )
            self.logstd_head = torch.nn.Sequential(
                torch.nn.Linear(16*self.num_nodes, 128),
                torch.nn.GELU(),
                torch.nn.LayerNorm(128),
                torch.nn.Linear(128, self.action_dim)
            )

        def forward(self, state):
            if len(state.shape) == 2: # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
            batch_size = state.shape[0]
            num_nodes = state.shape[1]
            num_features = state.shape[2]
            x = state.reshape(batch_size * num_nodes, num_features)
            batch_edge_index = []
            for i in range(batch_size):
                batch_edge_index.append(self.edge_index + i * num_nodes)
            batch_edge_index = torch.cat(batch_edge_index, dim=1).to(self.train_device)
            batch_edge_type = self.edge_type.repeat(batch_size).to(self.train_device)

            x = F.relu(self.conv1(x, batch_edge_index, batch_edge_type))
            x = F.relu(self.conv2(x, batch_edge_index, batch_edge_type)) 
            x = F.relu(self.conv3(x, batch_edge_index, batch_edge_type))  
            x = F.relu(self.conv4(x, batch_edge_index, batch_edge_type)) 
            x = x.reshape(batch_size, -1)

            mu = torch.tanh(self.mu_head(x))
            log_std = self.logstd_head(x).clamp(-20, 2)
            std = torch.exp(log_std)
            return mu, std

class ActorCriticRGCN:
    class Actor(torch.nn.Module): 
        def __init__(self, config: dict):
            super().__init__()    
            self.action_dim = config['action_dim']                 
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.edge_type = torch.tensor(graph['edge_type'], dtype=torch.long)
            self.num_nodes = graph['num_nodes']
            self.num_relations = graph['num_relations']           
    
            self.in_channels = self.num_node_features              
            self.out_channels = self.action_dim    
            self.conv1 = RGCNConv(self.in_channels, 32, self.num_relations, 
                                  num_bases=32)                    
            self.conv2 = RGCNConv(32, 32, self.num_relations,
                                  num_bases=32)                    
            self.conv3 = RGCNConv(32, 16, self.num_relations,
                                  num_bases=32)                    
            self.conv4 = RGCNConv(16, 16, self.num_relations,
                                  num_bases=32)                    
            self.lin1 = LazyLinear(self.out_channels)             
    
        def forward(self, state):  
            if len(state.shape) == 2: # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1]) 
    
            batch_size = state.shape[0]  
            edge_index = self.edge_index 
            edge_type = self.edge_type       
    
            actions = torch.tensor(()).to(self.train_device)    
            for i in range(batch_size):               
                x = state[i]                         
                x = F.relu(self.conv1(x, edge_index, edge_type))  
                x = F.relu(self.conv2(x, edge_index, edge_type))  
                x = F.relu(self.conv3(x, edge_index, edge_type))  
                x = F.relu(self.conv4(x, edge_index, edge_type))  
                x = self.lin1(torch.flatten(x))       
                x = torch.tanh(x).reshape(1, -1)     
                actions = torch.cat((actions, x), axis=0) 
    
            return actions      
    
    
    class Critic(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.action_dim = config['action_dim']              
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.edge_type = torch.tensor(graph['edge_type'], dtype=torch.long)
            self.num_nodes = graph['num_nodes']
            self.num_relations = graph['num_relations']
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.conv1 = RGCNConv(self.in_channels, 32, self.num_relations, 
                                  num_bases=32)
            self.conv2 = RGCNConv(32, 32, self.num_relations,
                                  num_bases=32)
            self.conv3 = RGCNConv(32, 16, self.num_relations,
                                  num_bases=32)
            self.conv4 = RGCNConv(16, 16, self.num_relations,
                                  num_bases=32)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            edge_index = self.edge_index
            edge_type = self.edge_type
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(  
                batch_size, self.num_nodes, -1)                            
            data = torch.cat((state, action), axis=2)                     
    
            values = torch.tensor(()).to(self.train_device)         
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.conv1(x, edge_index, edge_type))
                x = F.relu(self.conv2(x, edge_index, edge_type))
                x = F.relu(self.conv3(x, edge_index, edge_type))
                x = F.relu(self.conv4(x, edge_index, edge_type))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)    
            return values
        
class ActorCriticGCN:
    class Actor(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.action_dim = config['action_dim']
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.num_nodes = graph['num_nodes']
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.conv1 = GCNConv(self.in_channels, 32)
            self.conv2 = GCNConv(32, 32)
            self.conv3 = GCNConv(32, 16)
            self.conv4 = GCNConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
            edge_index = self.edge_index
    
            actions = torch.tensor(()).to(self.train_device)
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    class Critic(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.action_dim = config['action_dim']
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.num_nodes = graph['num_nodes']
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.conv1 = GCNConv(self.in_channels, 32)
            self.conv2 = GCNConv(32, 32)
            self.conv3 = GCNConv(32, 16)
            self.conv4 = GCNConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            edge_index = self.edge_index
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(self.train_device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values
        
class ActorCriticGAT:
    class Actor(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.action_dim = config['action_dim']
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.num_nodes = graph['num_nodes']
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.conv1 = GATConv(self.in_channels, 32)
            self.conv2 = GATConv(32, 32)
            self.conv3 = GATConv(32, 16)
            self.conv4 = GATConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
            edge_index = self.edge_index
    
            actions = torch.tensor(()).to(self.train_device)
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    class Critic(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.num_nodes = graph['num_nodes']
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.conv1 = GATConv(self.in_channels, 32)
            self.conv2 = GATConv(32, 32)
            self.conv3 = GATConv(32, 16)
            self.conv4 = GATConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            edge_index = self.edge_index
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(self.train_device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values
        
class ActorCriticMLP:
    class Actor(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.action_dim = config['action_dim']
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.num_nodes = graph['num_nodes'] 
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.mlp1 = Linear(self.in_channels, 32)
            self.mlp2 = Linear(32, 32)
            self.mlp3 = Linear(32, 16)  
            self.mlp4 = Linear(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
    
            actions = torch.tensor(()).to(self.train_device)    
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.mlp1(x))
                x = F.relu(self.mlp2(x))
                x = F.relu(self.mlp3(x))
                x = F.relu(self.mlp4(x))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    class Critic(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.action_dim = config['action_dim']
            self.train_device = config['train_device']   
            graph = config['graph']   
            self.num_node_features = graph['num_node_features']    
            self.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).T
            self.num_nodes = graph['num_nodes']     
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.mlp1 = Linear(self.in_channels, 32)
            self.mlp2 = Linear(32, 32)
            self.mlp3 = Linear(32, 16)
            self.mlp4 = Linear(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(self.train_device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.mlp1(x))
                x = F.relu(self.mlp2(x))
                x = F.relu(self.mlp3(x))
                x = F.relu(self.mlp4(x))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values
