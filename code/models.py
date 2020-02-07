import torch
from torch.nn import Linear, Dropout, BCEWithLogitsLoss, Softmax, ModuleList, LSTM, Embedding
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F

import copy

############################ MODEL CLASSES ############################

class GraphNet(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 graph_layers_sizes: int):
        
        super(GraphNet, self).__init__()
        
        graph_layers_sizes = copy.copy(graph_layers_sizes)
        graph_layers_sizes.insert(0,num_node_features)
        self.conv_layers = []
        for i in range(len(graph_layers_sizes) - 1):
            self.conv_layers.append(GCNConv(graph_layers_sizes[i], graph_layers_sizes[i+1]))
            self.conv_layers = ModuleList(self.conv_layers)

    def forward(self, data):
        
        x, edge_index, batch_vec = data.x, data.edge_index, data.batch

        for gcn_layer in self.conv_layers:
            x = gcn_layer(x, edge_index)
            x = F.relu(x)
            
        sum_vector = global_add_pool(x, batch = batch_vec)

        return sum_vector
    
class TextNet(torch.nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 output_dim: int, 
                 emb_dim: int):
        
        super(TextNet, self).__init__() 
        self.embedding = Embedding(vocab_size, emb_dim)
        self.rnn = LSTM(emb_dim, 
                        output_dim//2, 
                        num_layers=1, 
                        bidirectional=True,
                        batch_first=True)

    def forward(self, data):
        
        # get embedding and pass through LSTM
        emb = self.embedding(data.text)
        output, (hidden, cell) = self.rnn(emb)

        #concat the final forward and backward hidden layers and apply dropout
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
            
        return hidden.squeeze(0)

    
class FullModel(torch.nn.Module):
    def __init__(self, 
                 model_type: str,
                 num_classes: int,
                 num_node_features: int,  
                 graph_layers_sizes: list,
                 vocab_size: int,
                 text_embed_dim: int,
                 text_output_dim: int,
                 linear_layers_sizes: list,
                 dropout_rate: float):
        
        super(FullModel, self).__init__()
        
        self.model_type = model_type.lower()
        self.graph_net = GraphNet(num_node_features,graph_layers_sizes)
        self.text_net = TextNet(vocab_size,text_output_dim,text_embed_dim)
        self.dropout = Dropout(dropout_rate)
        
        if self.model_type == 'graph':
            linear_layer_input = graph_layers_sizes[-1]
        elif self.model_type == 'text':
            linear_layer_input = text_output_dim
        elif self.model_type == 'combo':
            linear_layer_input = graph_layers_sizes[-1]+text_output_dim
            
        linear_layers_sizes = copy.copy(linear_layers_sizes)
        linear_layers_sizes.insert(0,linear_layer_input)
        self.linear_layers = []
        for i in range(len(linear_layers_sizes) - 1):
            self.linear_layers.append(Linear(linear_layers_sizes[i], linear_layers_sizes[i+1]))
            self.linear_layers = ModuleList(self.linear_layers)
        self.predictor = Linear(linear_layers_sizes[-1], num_classes)

    def forward(self, data):
        
        if self.model_type == 'graph':
            x = self.graph_net(data)
        elif self.model_type == 'text':
            x = self.text_net(data)
        elif self.model_type == 'combo':
            text_vec = self.text_net(data)
            graph_vec = self.graph_net(data)
            if len(text_vec.size())==1:
                text_vec = text_vec.view(1,-1)  
            x = torch.cat([text_vec,graph_vec],1)

        for layer in self.linear_layers:
            x = self.dropout(F.relu(layer(x)))
        preds = self.predictor(x)
        return preds
    
    
class GoogleGraphNet(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 dropout_rate: float = 0.47):
        
        super(GoogleGraphNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 15, {'aggr':'max'})
        self.conv2 = GCNConv(15, 20, {'aggr':'max'})
        self.conv3 = GCNConv(20, 27, {'aggr':'max'})
        self.conv4 = GCNConv(27, 36, {'aggr':'max'})
        self.lin1 = Linear(36,96)
        self.lin2 = Linear(96,num_classes)
        self.dropout = Dropout(dropout_rate)
        self.softmax = Softmax(dim=1)

    def forward(self, data):
        
        x, edge_index, batch_vec = data.x, data.edge_index, data.batch

        c1 = self.conv1(x, edge_index)
        c1 = F.selu(c1)
        c2 = self.conv2(c1, edge_index)
        c2 = F.selu(c2)
        c3 = self.conv3(c2, edge_index)
        c3 = F.selu(c3)
        c4 = self.conv4(c3, edge_index)
        c4 = F.selu(c4)
        sum_vector = global_add_pool(c4, batch = batch_vec)
        x = self.softmax(sum_vector)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dropout(x, training=self.training)

        return x