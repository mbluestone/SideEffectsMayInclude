from pysmiles import read_smiles
import networkx as nx
import numpy as np
import pandas as pd
import torch
import sys
import os
from itertools import permutations

from skmultilearn.model_selection import iterative_train_test_split

from torch_geometric.data import Data, Dataset, DataLoader

def load_raw_data(path):
    '''
    Custom data loading function, depends on format of saved data
    '''
    
    data = pd.read_csv(path)
    X = data.smiles
    y = data.iloc[:,1:]
    
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
    
    # save is path is provided then save the split data separately
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

def generate_ngrams(x,n):
    n_grams = [' '.join(n_gram) for n_gram in list(zip(*[x[i:] for i in range(n)]))]
    return n_grams

def load_data_for_model_training(data_dir: str, 
                                 model_type: str,
                                 batch_size: int,
                                 ngram: int = 1):
    
    # get positive weights and labels
    train_labels_df = load_raw_data(path_join(data_dir,'train.csv'))[1]
    labels = train_labels_df.columns.tolist()
    pos_weight = get_pos_weights(np.matrix(train_labels_df))
    
    # if graph model
    if model_type in ['graph']:
        
        # load datasets
        datasets = {x: MoleculeDataset(load_raw_data(path_join(data_dir, x+'.csv')))
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
        
    # if nlp model
    elif model_type in ['nlp','bert']:
        
        tokenize = lambda x: generate_ngrams(x,ngram)
        TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
        LABEL = Field(sequential=False, use_vocab=False)

        tv_datafields = [("smiles", TEXT)]
        tv_datafields.extend([(label, LABEL) for label in labels])

        train, val = TabularDataset.splits(path=data_dir,
                                         train='train.csv', 
                                         validation="val.csv", 
                                         format='csv', 
                                         skip_header=True, 
                                         fields=tv_datafields)

        TEXT.build_vocab(trn)


        dataset_sizes = dict()
        dataset_sizes['train'] = len(train)
        dataset_sizes['val'] = len(val)

        train_iter, val_iter = BucketIterator.splits((train, val), 
                                                     batch_sizes=(30, 30),
                                                     device=device,
                                                     sort_key=lambda x: len(x.smiles),
                                                     sort_within_batch=False,
                                                     repeat=False)



        dataloaders=dict()
        dataloaders['train'] = TextBatchWrapper(train_iter, "smiles", 
                                                labels)
        dataloaders['val'] = TextBatchWrapper(val_iter, "smiles", 
                                              labels)
        
        # get number of features for the nodes
        num_node_features = 0
        
    return dataloaders, dataset_sizes, pos_weight, labels, num_node_features

class Molecule(Data):
    '''
    Molecule Data Class
    Subclass of torch-geometric.data.Data
    '''
    def __init__(self, smiles_string: str, y_list: list):
        """
        Args:
            smiles_string (string): SMILES fro the molecule.
            y_list (list): list of multilabels.
        """

        # create graph from smiles 
        # (the sys code is to block a pysmiles warning about iseometric stuff)
        sys.stdout = open(os.devnull, 'w')
        self.graph = read_smiles(smiles_string)
        sys.stdout = sys.__stdout__
        
        # inherit superclass from torch-geometric
        super().__init__(x = torch.tensor(self.extract_features(), 
                                          dtype=torch.float),
                         edge_index = torch.tensor(self.graph_to_edge_index(), 
                                                   dtype=torch.long),
                         y = torch.tensor(y_list, dtype=torch.float32))
        
        # remove graph attribute, necessary to inhereit from superclass
        del self.graph
    
    def extract_features(self):
        
        all_feature_vectors = np.array([])
        
        for atom_index in range(len(self.graph.nodes)):
            
            atom = self.graph.nodes[atom_index]
            
            feature_vector = self.convert_atom_to_vector(atom)
            
            if all_feature_vectors.size == 0:
                all_feature_vectors = feature_vector
            else:
                all_feature_vectors = np.vstack((all_feature_vectors, feature_vector))
            
        feature_matrix = np.matrix(all_feature_vectors)
        
        return feature_matrix
    
    def convert_atom_to_vector(self,atom):
        
        atom_dict = pd.read_csv('raw_data/atom_info.txt',sep=',').set_index('Symbol')['AtomicNumber'].to_dict()
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

    def __init__(self, Xy, transform=None):
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
        if isinstance(self.y, pd.DataFrame):
            self.labels = self.y.columns.tolist()
            self.y = np.matrix(self.y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        molecule = Molecule(self.X[idx],
                           self.y[idx].tolist())

        if self.transform:
            molecule = self.transform(molecule)

        return molecule
    
    # override unneccesary functions from super class
    def _download(self):
        pass
    def _process(self):
        pass
    
    def get(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        molecule = Molecule(self.X[idx],
                           self.y[idx].tolist())

        if self.transform:
            molecule = self.transform(molecule)

        return molecule

class TextDataObject(object):
    pass
    
class TextBatchWrapper:
    def __init__(self, dl, x_var, y_vars):
        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars
        
    def __iter__(self):
        for batch in self.dl:
            
            data = TextDataObject()
            data.x = getattr(batch, self.x_var)
            if self.y_vars is not None:
                data.y = torch.cat([getattr(batch, feat).unsqueeze(1) 
                                    for feat in self.y_vars], dim=1).float()
            else:
                data.y = torch.zeros((1))

            yield data

    def __len__(self):
        return len(self.dl)