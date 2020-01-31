import torch
from torch.nn import Linear, Dropout, BCEWithLogitsLoss, Softmax, ModuleList, LSTM, Embedding
from torch_geometric.nn import GCNConv, global_add_pool
import torch.nn.functional as F

############################ MODEL CLASSES ############################

class GraphNet(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 graph_layers_sizes: int):
        
        super(GraphNet, self).__init__()
        #self.conv1 = GCNConv(num_node_features, 16)
        #self.conv2 = GCNConv(16, 16)
        
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
    
class NLPNet(torch.nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 output_dim: int, 
                 emb_dim: int):
        
        super(NLPNet, self).__init__() 
        self.embedding = Embedding(vocab_size, emb_dim)
        self.rnn = LSTM(emb_dim, 
                        output_dim//2, 
                        num_layers=1, 
                        bidirectional=True)

    def forward(self, data):
        
        text = data.x
        
        output, (hidden, cell) = self.rnn(self.embedding(text))

        #concat the final forward and backward hidden layers and apply dropout
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
            
        return hidden.squeeze(0)

        #hdn, _ = self.rnn(self.embedding(x))
        #feature = hdn[-1, :, :]
        #return feature
    
class FullModel(torch.nn.Module):
    def __init__(self, 
                 model_type: str,
                 num_classes: int,
                 num_node_features: int,  
                 graph_layers_sizes: list,
                 vocab_size: int,
                 num_lstm_layers: int, 
                 nlp_embed_dim: int,
                 nlp_output_dim: int,
                 linear_layers_sizes: list,
                 dropout_rate: float):
        
        super(FullModel, self).__init__()
        
        self.model_type = model_type.lower()
        self.graph_net = GraphNet(num_node_features,graph_layers_sizes)
        self.nlp_net = NLPNet(vocab_size,nlp_output_dim,nlp_embed_dim)
        self.dropout = Dropout(dropout_rate)
        
        if self.model_type == 'graph':
            linear_layer_input = graph_layers_sizes[-1]
        elif self.model_type == 'nlp':
            linear_layer_input = nlp_output_dim
        elif self.model_type == 'combo':
            linear_layer_input = graph_layers_sizes[-1]+nlp_output_dim
            
        linear_layers_sizes.insert(0,linear_layer_input)
        self.linear_layers = []
        for i in range(len(linear_layers_sizes) - 1):
            self.linear_layers.append(Linear(linear_layers_sizes[i], linear_layers_sizes[i+1]))
            self.linear_layers = ModuleList(self.linear_layers)
        self.predictor = Linear(linear_layers_sizes[-1], num_classes)

    def forward(self, data):
        
        if self.model_type == 'graph':
            x = self.graph_net(data)
        elif self.model_type == 'nlp':
            x = self.nlp_net(data)
        elif self.model_type == 'nlp':
            x = self.nlp_net(data)

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