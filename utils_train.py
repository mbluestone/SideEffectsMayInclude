# MolNet
# Max Bluestone

# Using a graph/NLP model to train and test.

import torch
from torch.nn import Linear, BCEWithLogitsLoss
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.transforms import AddSelfLoops, ToDense

import pandas as pd
import random
import time
import csv

from utils_data import *

from os import mkdir
from os.path import join as path_join
from os.path import dirname

from sklearn.metrics import hamming_loss, recall_score, precision_score, f1_score, confusion_matrix


#### MODEL CLASS #####
class MoleculeNet(torch.nn.Module):
    def __init__(self, 
                 num_node_features: int, 
                 num_classes: int, 
                 num_graph_layers: int,
                 num_linear_layers: int):
        
        super(MoleculeNet, self).__init__()
        self.num_node_features = num_node_features
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = Linear(16,100)
        self.lin2 = Linear(100,num_classes)

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


def create_model(model_type: str,
                 num_node_features: int,
                 num_classes: int,
                 num_graph_layers: int,
                 num_linear_layers: int, 
                 device, 
                 pretrain_load_path: str = None) -> MoleculeNet:
    """
    Instantiate the model.
    Args:
        num_graph_layers: Number of layers to use in the model from [18, 34, 50, 101, 152].
        num_classes: Number of classes in the dataset.
        pretrain_load_path: Use pretrained weights.
    Returns:
        The instantiated model with the requested parameters.
    """

    # make sure a correct model type is requested
    possible_models = ['graph', 'nlp', 'bert', 'combo']
    assert model_type.lower() in possible_models, f"Model type must be one of {possible_models} not {model_type}"
    
    # if only graph model is desired
    if model_type.lower() == 'graph':
        model = MoleculeNet(num_node_features, 
                            num_classes, 
                            num_graph_layers, 
                            num_linear_layers).to(device)
        
    # if only BERT model is desired
    if model_type.lower() == 'bert':
        model = MoleculeNet(num_node_features, 
                            num_classes, 
                            num_graph_layers, 
                            num_linear_layers).to(device)
        
    # if combo model is desired
    if model_type.lower() == 'combo':
        model = MoleculeNet(num_node_features, 
                            num_classes, 
                            num_graph_layers, 
                            num_linear_layers).to(device)
    

    # if loading a pretrained model from a state dict
    if pretrain_load_path:
        model.load_state_dict(torch.load(pretrain_load_path, map_location=device))
        
    # transfer model to cpu or gpu
    model = model.to(device=device)
        
    return model

def get_pos_weights(labels):
    '''
    Calculate the positive weights for each class
    '''
    weights = [(labels.shape[0]-labels[:,i].sum())/labels[:,i].sum() 
               for i in range(labels.shape[1])]
    return torch.tensor(weights)

def train_model(data_dir: str,
                model_type: str,
                num_epochs: int,
                num_graph_layers: int,
                num_linear_layers: int,
                learning_rate: int,
                learning_rate_decay: int,
                weight_decay: int, 
                batch_size: int,
                log_csv: str,
                log_file: str = None,
                pretrain_load_path: str = None):
    '''
    Function for training model
    
    Args:
        num_epochs
    '''    
    
    # load datasets
    datasets = {x: MoleculeDataset(load_raw_data(path_join(data_dir, x+'.csv')),
                                     transform=AddSelfLoops())
                for x in ['train', 'val']}
    
    # get labels and make sure train and val data have the same
    labels = datasets['train'].labels
    assert labels == datasets['val'].labels
    
    # get number of features for the nodes
    num_node_features = len(datasets['train'][0].x[0])
    
    # create dataloaders
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) 
                   for x in ['train', 'val']}
    
    # get the size of each dataset
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"num labels: {len(labels)}\n"
          f"num train molecules {dataset_sizes['train']}\n"
          f"num val molecules {dataset_sizes['val']}\n"
          f"CUDA is_available: {torch.cuda.is_available()}")
    
    # instantiate model
    model = create_model(model_type=model_type,
                         num_node_features=num_node_features,
                         num_classes=len(labels),
                         num_graph_layers=num_graph_layers,
                         num_linear_layers=num_linear_layers,
                         pretrain_load_path=pretrain_load_path,
                         device=device)
    
    # instantiate optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # learning rate: exponential
    scheduler = lr_scheduler.ExponentialLR(optimizer,
                                           gamma=learning_rate_decay)

    # initialize loss function
    criterion = BCEWithLogitsLoss(pos_weight=get_pos_weights(datasets['train'].y))

    # create the logging csv
    if not os.path.exists(dirname(log_csv)):
        os.mkdir(dirname(log_csv))
    with open(log_csv,'w') as file:
        writer = csv.writer(file)
        
        # write header
        writer.writerow(['epoch','train_loss','train_acc','train_f1','val_loss','val_acc','val_f1'])
        
        # train the model
        train_helper(model=model,
                     device=device,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     labels=labels, 
                     num_epochs=num_epochs, 
                     dataloaders=dataloaders,
                     dataset_sizes=dataset_sizes,
                     criterion=criterion, 
                     log_file=log_file, 
                     log_csv=log_csv,
                     writer=writer)
    

def train_helper(model: torch.nn.Module,
                 device: torch.device,
                 optimizer,
                 scheduler,
                 labels: list, 
                 num_epochs: int, 
                 dataloaders: dict,
                 dataset_sizes: dict,
                 criterion: torch.nn.modules.loss, 
                 log_file: str, 
                 log_csv: str,
                 writer):
    '''
    Helper function for training model
    
    Args:
        model: torch.nn.Module, 
        labels: list, 
        num_epochs: int, 
        dataloaders: dict,
        dataset_sizes: dict,
        criterion: , 
        log_file: str, 
        log_csv: str
    '''

    # start tracking time
    start = time.time()
    
    # loop through epochs
    for epoch in range(num_epochs):

        print(f'Epoch {epoch}:')
        
        # Training
        model.train()

        # initialize running loss and accuracy for the epoch
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        train_running_precision = 0.0
        train_running_recall = 0.0
        train_running_f1 = 0.0

        # loop through batched training data
        for inputs in dataloaders['train']:
            
            # pull out batch labels
            train_batch_labels = inputs.y.numpy()
            
            # send to device
            inputs = inputs.to(device)
            print(inputs)
            print(inputs.y)

            # 
            optimizer.zero_grad()
            with torch.set_grad_enabled(mode=True):
                
                # make predicitions
                out = model(inputs)
                print(out)
                
                # calculate loss
                train_loss = criterion(out, inputs.y)
                
                train_batch_predictions = (torch.sigmoid(out)>0.5).cpu().numpy()
                
                # backpropagate
                train_loss.backward()
                
                # step optimizer
                optimizer.step()
                
                # calculate performance metrics
                train_acc = 1-hamming_loss(train_batch_labels,train_batch_predictions)
                train_precision = precision_score(train_batch_labels,train_batch_predictions,average='micro')
                train_recall = recall_score(train_batch_labels,train_batch_predictions,average='micro')
                train_f1 = f1_score(train_batch_labels,train_batch_predictions,average='micro')
                
            # update running metrics
            train_running_loss += train_loss.item() * inputs.y.size(0)
            train_running_accuracy += train_acc * inputs.y.size(0)
            train_running_precision += train_precision * inputs.y.size(0)
            train_running_recall += train_recall * inputs.y.size(0)
            train_running_f1 += train_f1 * inputs.y.size(0)

        # calculate training metrics for the epoch
        epoch_train_loss = np.round(train_running_loss/dataset_sizes['train'],
                                  decimals=4)
        epoch_train_acc = np.round(train_running_accuracy/dataset_sizes['train'],
                                  decimals=4)
        epoch_train_precision = np.round(train_running_precision/dataset_sizes['train'],
                                  decimals=4)
        epoch_train_recall = np.round(train_running_recall/dataset_sizes['train'],
                                  decimals=4)
        epoch_train_f1 = np.round(train_running_f1/dataset_sizes['train'],
                                  decimals=4)

        print(f'Training: Loss = {epoch_train_loss}, ' 
              f'Accuracy = {epoch_train_acc}, '
              f'Precision = {epoch_train_precision}, '
              f'Recall = {epoch_train_recall}, '
              f'F1 = {epoch_train_f1}')   

        # Validation
        model.eval()

        # initialize running loss and accuracy for the epoch
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        val_running_precision = 0.0
        val_running_recall = 0.0
        val_running_f1 = 0.0

        # loop through batched validation data
        for inputs in dataloaders['val']:
            
            # pull out batch labels
            val_batch_labels = inputs.y.numpy()
            
            # send to device
            inputs = inputs.to(device)

            with torch.set_grad_enabled(mode=False):
                
                # make predicitions
                out = model(inputs)
                
                # calculate loss
                val_loss = criterion(out, inputs.y)
                
                val_batch_predictions = (torch.sigmoid(out)>0.5).cpu().numpy()
                
                # calculate performance metrics
                val_acc = 1-hamming_loss(val_batch_labels,val_batch_predictions)
                val_precision = precision_score(val_batch_labels,val_batch_predictions,average='micro')
                val_recall = recall_score(val_batch_labels,val_batch_predictions,average='micro')
                val_f1 = f1_score(val_batch_labels,val_batch_predictions,average='micro')

            # update running metrics
            val_running_loss += val_loss.item() * inputs.y.size(0)
            val_running_accuracy += val_acc * inputs.y.size(0)
            val_running_precision += val_precision * inputs.y.size(0)
            val_running_recall += val_recall * inputs.y.size(0)
            val_running_f1 += val_f1 * inputs.y.size(0)

        # calculate validation metrics for the epoch
        epoch_val_loss = np.round(val_running_loss/dataset_sizes['val'],
                                  decimals=4)
        epoch_val_acc = np.round(val_running_accuracy/dataset_sizes['val'],
                                  decimals=4)
        epoch_val_precision = np.round(val_running_precision/dataset_sizes['val'],
                                  decimals=4)
        epoch_val_recall = np.round(val_running_recall/dataset_sizes['val'],
                                  decimals=4)
        epoch_val_f1 = np.round(val_running_f1/dataset_sizes['val'],
                                  decimals=4)
        
        # empty cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # step scheduler
        scheduler.step()

        print(f'Validation: Loss = {epoch_val_loss}, ' 
              f'Accuracy = {epoch_val_acc}, '
              f'Precision = {epoch_val_precision}, '
              f'Recall = {epoch_val_recall}, '
              f'F1 = {epoch_val_f1}')    
        
        # log metrics in log csv
        writer.writerow('{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f}\n'.format(
            str(epoch), epoch_train_loss, epoch_train_acc, epoch_train_f1,
            epoch_val_loss, epoch_val_acc, epoch_val_f1).split(','))

    #writer.close()
    
    # Print training information at the end.
    print(f"\nTraining complete in "
          f"{(time.time() - start) // 60:.2f} minutes")