import config
import os
import csv
import torch
from utils_data import *
from utils_train import *

from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim import lr_scheduler

def train_model():

    '''
    Function for training the model
    
    Args:
        None, pulls from config to get parameters
    '''    
    
    # create a dictionary of the model training parameters
    model_params_dict = get_parameters(config)
    
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load data objects
    dataloaders,dataset_sizes,pos_weight,labels,num_node_features,vocab_size = load_data_for_model(model_params_dict,device,training=True)
    
    print(f"num labels: {len(labels)}\n"
          f"num train molecules {dataset_sizes['train']}\n"
          f"num val molecules {dataset_sizes['val']}\n"
          f"CUDA is_available: {torch.cuda.is_available()}")
    
    # add new parameters to params dict
    model_params_dict['vocab_size'] = vocab_size
    model_params_dict['num_node_features'] = num_node_features
    model_params_dict['num_classes'] = len(labels)
    
    # instantiate model
    model = create_model(model_params_dict=model_params_dict,
                         device=device)
    
    # instantiate optimizer
    optimizer = Adam(model.parameters(), 
                     lr=model_params_dict['learning_rate'], 
                     weight_decay=model_params_dict['weight_decay'])
    
    # learning rate: exponential
    scheduler = lr_scheduler.ExponentialLR(optimizer,
                                           gamma=model_params_dict['learning_rate_decay'])

    # initialize loss function
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = criterion.to(device)

    # create the logging csv
    if not os.path.exists(dirname(model_params_dict['log_csv'])):
        os.mkdir(dirname(model_params_dict['log_csv']))
    with open(model_params_dict['log_csv'],'w') as file:
        writer = csv.writer(file)
        
        # write header
        writer.writerow(['epoch','train_loss','train_acc','train_f1',
                         'train_roc_auc','val_loss','val_acc','val_f1',
                         'val_roc_auc'])
        
        # train the model
        train_helper(model=model,
                     device=device,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     labels=labels,  
                     dataloaders=dataloaders,
                     dataset_sizes=dataset_sizes,
                     criterion=criterion, 
                     writer=writer,
                     model_params_dict=model_params_dict) 

        
if __name__ == '__main__':
    
    train_model()