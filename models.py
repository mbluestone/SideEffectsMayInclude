import torch
from torch.nn import Linear, Dropout, BCEWithLogitsLoss, Softmax
import torch.nn.functional as F

################################## MODEL CLASSES ##################################
class GraphNet(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 num_classes: int, 
                 num_graph_layers: int,
                 graph_layers_sizes: int):
        
        super(MoleculeNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.final_conv = GCNConv(16, num_classes)
        self.lin1 = Linear(16,100)
        self.lin2 = Linear(100,num_classes)
        self.dropout = Dropout(dropout_rate)

    def forward(self, data):
        
        x, edge_index, batch_vec = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        sum_vector = global_add_pool(x, batch = batch_vec)
        x = F.relu(sum_vector)
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        return x
    
class GoogleGraphNet(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 num_classes: int, 
                 num_graph_layers: int,
                 num_linear_layers: int,
                 dropout_rate: float):
        
        super(GoogleMoleculeNet, self).__init__()
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
        #print('Post sum:',sum_vector.size())
        x = self.softmax(sum_vector)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return x
    
class NLPNet(torch.nn.Module):
    def __init__(self, num_classes, hidden_dim, emb_dim=100, num_linear=1):
        super().__init__() 
        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)
        self.rnn = nn.LSTM(emb_dim, 
                           hidden_dim, 
                           num_layers=1, 
                           bidirectional=True)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x = data.x
        hdn, _ = self.encoder(self.embedding(x))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
            feature = layer(feature)
        preds = self.predictor(feature)
        return preds
    
class FCLinear(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 num_classes: int, 
                 num_graph_layers: int,
                 graph_layers_sizes: int):
        
        super(MoleculeNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.final_conv = GCNConv(16, num_classes)
        self.lin1 = Linear(16,100)
        self.lin2 = Linear(100,num_classes)
        self.dropout = Dropout(dropout_rate)

    def forward(self, data):
        
        x, edge_index, batch_vec = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        sum_vector = global_add_pool(x, batch = batch_vec)
        x = F.relu(sum_vector)
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        return x