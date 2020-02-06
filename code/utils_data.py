import numpy as np
import pandas as pd
import sys
import os
from os.path import join as path_join
from itertools import permutations
import dill

from pysmiles import read_smiles
import networkx as nx

from skmultilearn.model_selection import iterative_train_test_split

import torch

from torch_geometric.data import Data, Dataset, DataLoader

from torchtext.data import Field, TabularDataset, Iterator, BucketIterator

path_to_atom_info = 'raw_data/atom_info.txt'

def load_raw_data(path,label=None):
    '''
    Custom data loading function specific for my preprocessed dataset, 
    depends on format of saved data
    '''
    
    data = pd.read_csv(path)
    X = data.smiles
    y = data.iloc[:,1:]
    
    if label:
        y = y.loc[:,label]
    
    return X, y

def split_data(X, y, test_split = 0.2, val_split = 0.1, save_path = None):
    '''
    Function to split the data into train/val/test datasets
    '''
    
    # handle X input and extract column names if possible
    if isinstance(X, pd.DataFrame): # if dataframe
        X_column_names = X.columns
        X = np.matrix(X)  
    elif isinstance(X, pd.Series): # if series
        X_column_names = [X.name]
        X = np.matrix(X.values).T
    elif isinstance(X, np.matrix): # if matrix
        X_column_names = None
    elif not isinstance(X, np.matrix): # if neither
        raise Exception(f'X must be an object of type np.matrix or pd.DataFrame and not {type(X)}')
        
    # handle y input and extract column names if possible
    if isinstance(y, pd.DataFrame): # if dataframe
        y_column_names = y.columns
        y = np.matrix(y) 
    elif isinstance(y, pd.Series): # if series
        y_column_names = [y.name]
        y = np.matrix(y.values).T
    elif isinstance(X, np.matrix): # if matrix
        y_column_names = None
    elif not isinstance(y, np.matrix): # if neither
        raise Exception(f'y must be an object of type np.matrix or pd.DataFrame and not {type(y)}')
        
    # make sure X and y are the same size
    assert y.shape[0] == X.shape[0], f'X and y do not have the same number of samples, {X.shape[0]} and {y.shape[0]}'
    
    # split train/val and test data
    X, y, X_test, y_test = iterative_train_test_split(X,y,test_size = test_split)
    
    # split train and val data
    X_train, y_train, X_val, y_val = iterative_train_test_split(X,y,test_size = val_split)
    
    # convert back to dataframe
    X_train = pd.DataFrame(X_train,columns=X_column_names)
    y_train = pd.DataFrame(y_train,columns=y_column_names)
    X_val = pd.DataFrame(X_val,columns=X_column_names)
    y_val = pd.DataFrame(y_val,columns=y_column_names)
    X_test = pd.DataFrame(X_test,columns=X_column_names)
    y_test = pd.DataFrame(y_test,columns=y_column_names)
    
    # if save path is provided then save the split data separately
    if save_path:
    
        # save train data
        pd.concat([X_train,y_train],axis=1).to_csv(save_path+'train.csv',index=False)
        # save val data
        pd.concat([X_val,y_val],axis=1).to_csv(save_path+'val.csv',index=False)
        # save test data
        pd.concat([X_test,y_test],axis=1).to_csv(save_path+'test.csv',index=False)
     
    # just return the split data and don't save
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
    
def get_pos_weights(labels):
    '''
    Calculate the positive weights for each class
    '''
    weights = [(labels.shape[0]-labels[:,i].sum())/labels[:,i].sum() 
               for i in range(labels.shape[1])]
    return torch.tensor(weights)

def generate_bigrams(x):
    bi_grams = [''.join(bi_grams) for bi_grams in list(zip(*[x[i:] for i in range(2)]))]
    return bi_grams

def smiles_to_tensor(smiles,
                     stoi_dict):
    smiles_tensor = torch.tensor([stoi_dict[s] for s in smiles],
                                dtype=torch.long)
    return smiles_tensor

def pad_tensor(smiles_tensor,
               max_length):
    if len(smiles_tensor)>max_length:
        padded_tensor = smiles_tensor[:max_length]
    elif len(smiles_tensor)<max_length:
        padded_tensor = torch.cat([smiles_tensor,torch.ones(max_length-len(smiles_tensor), 
                                                           dtype=torch.long)])
    else:
        padded_tensor = smiles_tensor
    return padded_tensor

def process_smiles_for_nlp(smiles,
                           stoi_dict,
                           max_length):
    
    bigram_smiles = generate_bigrams(smiles.lower())
    tensor_smiles = smiles_to_tensor(bigram_smiles,stoi_dict)
    padded_smiles = pad_tensor(tensor_smiles,max_length)
    
    return padded_smiles

def get_graph_and_text_data(model_params_dict: dict,
                            labels: list,
                            training: bool):
    
    if training:
        phase_names = ['train','val']
    else:
        phase_names = ['test']
    
    # get text vocab
    if training:
        tokenizer = lambda x: generate_bigrams(x)
        TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = Field(sequential=False, use_vocab=False)

        datafields = [("smiles", TEXT)]
        datafields.extend([(label, LABEL) for label in labels])
        
        train = TabularDataset(path=model_params_dict['data_dir']+'train.csv',
                               format='csv',
                               skip_header=True,
                               fields=datafields)

        TEXT.build_vocab(train)
        
        
    # if testing
    else:
        
        # load TEXT field from 
        with open("trained_models/TEXT.Field","rb")as f:
             TEXT=dill.load(f)
                
    # get size of vocabulary            
    vocab_size = len(TEXT.vocab)
        
    # load datasets
    datasets = {x: MoleculeDataset(Xy=load_raw_data(path_join(model_params_dict['data_dir'], x+'.csv'),
                                                    label=model_params_dict['label']),
                                   stoi_dict=TEXT.vocab.stoi,
                                   max_text_length=200)
                for x in phase_names}

    # get number of features for the nodes
    num_node_features = len(datasets[phase_names[0]][0].x[0])
        
    # create dataloaders
    dataloaders = {x: DataLoader(datasets[x], batch_size=model_params_dict['batch_size'], shuffle=True, num_workers=4) 
                   for x in phase_names}

    # get the size of each dataset
    dataset_sizes = {x: len(datasets[x]) for x in phase_names}
    
    return dataloaders, dataset_sizes, num_node_features, vocab_size

def load_data_for_model(model_params_dict: dict, 
                        device: torch.device, 
                        training: bool):
    
    # get positive weights and labels
    if training:
        labels_df = load_raw_data(path_join(model_params_dict['data_dir'],'train.csv'), 
                                  label=model_params_dict['label'])[1]
    else:
        labels_df = load_raw_data(path_join(model_params_dict['data_dir'],'test.csv'), 
                                  label=model_params_dict['label'])[1]
        
    if isinstance(labels_df,pd.DataFrame):
        labels = labels_df.columns.tolist()
        pos_weight = get_pos_weights(np.matrix(labels_df))
    else: 
        labels = [labels_df.name]
        pos_weight = get_pos_weights(np.matrix(labels_df).reshape(-1,1))
    
    # load and process data for model
    dataloaders, dataset_sizes, num_node_features, vocab_size = get_graph_and_text_data(model_params_dict, 
                                                                                                 labels, 
                                                                                                 training)
        
    return dataloaders, dataset_sizes, pos_weight, labels, num_node_features, vocab_size

class Molecule(Data):
    '''
    Molecule Data Class
    Subclass of torch-geometric.data.Data
    '''
    def __init__(self, smiles_string: str, y_list: list):
        """
        Args:
            smiles_string (string): SMILES for the molecule.
            y_list (list): list of multilabels.
        """

        # create graph from smiles 
        # (the sys code is to block a pysmiles warning about iseometric stuff)
        sys.stdout = open(os.devnull, 'w')
        self.graph = read_smiles(smiles_string)
        sys.stdout = sys.__stdout__
        
        if isinstance(y_list,list):
            y = torch.tensor(y_list, dtype=torch.float32)
        else:
            y = torch.tensor(y_list, dtype=torch.float32).view(1,-1)
        
        # inherit superclass from torch-geometric
        super().__init__(x = torch.tensor(self.extract_features(), 
                                          dtype=torch.float),
                         edge_index = torch.tensor(self.graph_to_edge_index(), 
                                                   dtype=torch.long),
                         y = y)
        
        # remove graph attribute, necessary to inherit from superclass
        del self.graph
    
    def extract_features(self):
        
        all_feature_vectors = np.array([])
        
        for atom_index in [node for node,data in self.graph.nodes.items()]:
            
            atom = self.graph.nodes[atom_index]
            
            feature_vector = self.convert_atom_to_vector(atom)
            
            if all_feature_vectors.size == 0:
                all_feature_vectors = feature_vector
            else:
                all_feature_vectors = np.vstack((all_feature_vectors, feature_vector))
            
        feature_matrix = np.matrix(all_feature_vectors)
        
        return feature_matrix
    
    def convert_atom_to_vector(self,atom):
        
        atom_dict = pd.read_csv(path_to_atom_info,sep=',').set_index('Symbol')['AtomicNumber'].to_dict()
        feature_vector = np.array([])
        feature_vector = np.append(feature_vector,int(atom_dict[atom['element']]))
        feature_vector = np.append(feature_vector,atom['charge'])
        if atom['aromatic']:
            feature_vector = np.append(feature_vector,1)
        else:
            feature_vector = np.append(feature_vector,0)
        feature_vector = np.append(feature_vector,atom['hcount'])
        
        return feature_vector
    
    def graph_to_edge_index(self):
        '''
        Convert the adjacency matrix to an edge index for representing
        graph connectivity in COO format
        '''
        adjacency_matrix = nx.to_numpy_matrix(self.graph, weight='order')

        edge_index = np.array([])
        for i,j in permutations(range(len(self.graph.nodes)),2):
            if adjacency_matrix[i,j] != 0:
                if edge_index.size == 0:
                    edge_index = np.array([i,j])
                else:
                    edge_index = np.vstack((edge_index, np.array([i,j])))
                        
        return edge_index.T
        
class MoleculeDataset():
    """Molecules dataset."""

    def __init__(self, 
                 Xy, 
                 transform=None, 
                 stoi_dict=None,
                 max_text_length=None):
        """
        Args:
            X (NumPy matrix or Pandas DataFrame, n_samples*n_features): SMILES data.
            y (NumPy matrix or Pandas DataFrame, n_samples*n_labels): multilabel classifications.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = Xy[0]
        self.y = Xy[1]
        self.transform = transform
        self.stoi_dict = stoi_dict
        self.max_text_length = max_text_length
        if isinstance(self.y, pd.DataFrame):
            self.labels = self.y.columns.tolist()
            self.y = np.matrix(self.y)
        if isinstance(self.y, pd.Series):
            self.labels = self.y.name

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        molecule = Molecule(self.X[idx],
                           self.y[idx].tolist())

        if self.transform:
            molecule = self.transform(molecule)
            
        molecule.text = process_smiles_for_nlp(self.X[idx], 
                                               self.stoi_dict, 
                                               self.max_text_length).view(1,-1)
        
        molecule.smiles = self.X[idx]

        return molecule
    
    # override unneccesary functions from super class
    def _download(self):
        pass
    def _process(self):
        pass
    
    