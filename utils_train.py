import pandas as pd
import torch
from torch.nn import Linear, BCEWithLogitsLoss
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.transforms import AddSelfLoops, ToDense
from data import MoleculeDataset
import random

from utils_data import *

from os.path import join as path_join

from sklearn.metrics import accuracy_score, recall, precision, f1_score, confusion_matrix

#### DATA #####
#load
data_path = '~/github/MolNet/raw_data/sider_data/sider.csv'
sider_data = pd.read_csv(data_path)
smiles_list = sider_data.smiles.to_list()
dataset = MoleculeDataset(data_path, transform=AddSelfLoops())

train_path = '~/github/MolNet/raw_data/sider_data/train_sider.csv'
val_path = '~/github/MolNet/raw_data/sider_data/val_sider.csv'
test_path = '~/github/MolNet/raw_data/sider_data/test_sider.csv'

train_dataloader = DataLoader(train_dataset,batch_size=50,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=50,shuffle=True)


class_names = image_datasets['train'].classes


#### MODEL CLASS #####
class MoleculeNet(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 num_classes: int, 
                 num_graph_layers: int,
                 num_linear_layers: int):
        
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = Linear(16,100)
        self.lin2 = Linear(100,num_classes)
        self.sig = Sigmoid()

    def forward(self, data):
        
        x, edge_index, batch_vec = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        sum_vector = global_add_pool(x, batch = batch_vec)
        x = F.relu(sum_vector)
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        return x


def create_model(num_graph_layers: int,
                 num_linear_layers: int, 
                 num_classes: int,
                 pretrain_load_path=None: str) -> MoleculeNet:
    """
    Instantiate the model.
    Args:
        num_graph_layers: Number of layers to use in the model from [18, 34, 50, 101, 152].
        num_classes: Number of classes in the dataset.
        pretrain_load_path: Use pretrained weights.
    Returns:
        The instantiated model with the requested parameters.
    """

    model = MoleculeNet(num_node_features, num_classes).to(device)

    if pretrain:
        pretrained = model_constructor(pretrained=True).state_dict()
        if num_classes != pretrained["fc.weight"].size(0):
            del pretrained["fc.weight"], pretrained["fc.bias"]
        model.load_state_dict(state_dict=pretrained, strict=False)
    return model

def get_pos_weights(labels):
    '''
    Calculate the positive weights for each class
    '''
    weights = [(labels.shape[0]-labels[:,i].sum())/labels[:,i].sum() 
               for i in range(labels.shape[1])]
    return torch.tensor(weights)

def train_model(data_dir: str,
                labels: list,
                num_epochs: int,
                num_graph_layers: int,
                num_linear_layers: int,
                learning_rate: int,
                weight_decay: int, 
                batch_size: int):
    '''
    Function for training model
    
    Args:
        num_epochs
    '''
    
    # load data
    
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_node_features = 4
    
    
    # load datasets
    datasets = {x: MoleculeDataset(load_raw_data(path_join(data_dir, x)),
                                     transform=AddSelfLoops())
                for x in ['train', 'val']}
    
    # create dataloaders
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) 
                   for x in ['train', 'val']}
    
    # get the size of each dataset
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    
    datasets['train'][0].y 
    
    # 
    model = create_model(num_node_features, num_classes)
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # initialize loss function
    criterion = BCEWithLogitsLoss(pos_weights=get_pos_weights())
    
    
    pos_weights = get_pos_weights()
    
    train_helper(model, num_epochs, dataloaders, criterion)
    

def train_helper(model, num_epochs, dataloaders, criterion):
    '''
    Helper function for training model
    
    Args:
        model:
        num_epochs
    '''

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    all_metrics = []
    for epoch in range(num_epochs):

        print(f'Epoch {epoch}:')
        model.train()

        train_running_loss = 0.0
        train_running_accuracy = 0.0

        for inputs in train_dataloader:
            
            batch_labels = inputs.y.numpy()

            optimizer.zero_grad()
            with torch.set_grad_enabled(mode=True):
                
                # make predicitions
                out = model(inputs)
                batch_predictions = (torch.sigmoid(out)>0.5).numpy()
                
                # calculate loss
                train_loss = criterion(out, inputs.y)
                
                # backpropagate
                train_loss.backward()
                
                # 
                optimizer.step()
                
                # calculate performance metrics
                train_acc = accuracy_score(inputs.y.numpy(),)
                train_precision = precision(inputs.y.numpy(),(torch.sigmoid(out)>0.5).numpy())
                train_recall = recall(inputs.y.numpy(),(torch.sigmoid(out)>0.5).numpy())
                train_f1 = f1_score(inputs.y.numpy(),(torch.sigmoid(out)>0.5).numpy())
                

            train_running_loss += train_loss.item() * inputs.y.size(0)
            train_running_accuracy += train_acc * inputs.y.size(0)

        epoch_train_loss = train_running_loss/len(train_dataset)
        train_losses.append(epoch_train_loss)
        epoch_train_acc = train_running_accuracy/len(train_dataset)
        train_accuracies.append(epoch_train_acc)

        print(f'Train: Loss = {epoch_train_loss}, Acc = {epoch_train_acc}')   

        model.eval()

        val_running_loss = 0.0
        val_running_accuracy = 0.0

        for inputs in val_dataloader:

            with torch.set_grad_enabled(mode=False):
                out = model(inputs)
                val_loss = criterion(out, inputs.y)
                val_acc = ((torch.sigmoid(out)>0.5).numpy()==inputs.y.numpy()).mean()

            val_running_loss += val_loss.item() * inputs.y.size(0)
            val_running_accuracy += val_acc * inputs.y.size(0)

        epoch_val_loss = val_running_loss/len(val_dataset)
        val_losses.append(epoch_val_loss)
        epoch_val_acc = val_running_accuracy/len(val_dataset)
        val_accuracies.append(epoch_val_acc) 

        print(f'Validation: Loss = {epoch_val_loss}, ' 
              f'Acc = {epoch_val_acc}'
              f'Precision = {val_precision}')