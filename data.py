from pysmiles import read_smiles
import networkx as nx
import numpy as np
import pandas as pd
import torch
import sys
import os

from skmultilearn.model_selection import iterative_train_test_split

from torch_geometric.data import Data, Dataset, DataLoader

def load_data(path):
    '''
    Custom data loading function, depends on format of saved data
    '''
    
    data = pd.read_csv(path)
    X = np.matrix(data.smiles.values).T
    y = np.matrix(data.iloc[:,1:].values)
    
    return X, y

def split_data(X, y, test_size = 0.2, save_path = None):
    
    X_train, y_train, X_test, y_test = iterative_train_test_split(X,y,test_size = 0.2)
    
    train_data = pd.concat(X_train,y_train)
    val_data = pd.concat(X_train,y_train)
    test_data = pd.concat(X_train,y_train)

class Molecule(Data):
    '''
    Molecule Data Class
    Subclass of torch-geometric.data.Data
    '''
    def __init__(self, smiles_string: str, y_list: list):

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
    
    #def __len__(self):
     #   return len(self.graph.nodes)
    
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
        #print(adjacency_matrix)
        edge_index = np.array([])
        for i in range(len(self.graph.nodes)):
            for j in range(len(self.graph.nodes)):
                if adjacency_matrix[i,j] != 0:
                    if edge_index.size == 0:
                        edge_index = np.array([i,j])
                    else:
                        edge_index = np.vstack((edge_index, np.array([i,j])))
                        
        return edge_index.T
                    
            
        
class MoleculeDataset():
    """Molecules dataset."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            smiles_list (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        molecule = Molecule(self.X[idx],
                           self.y[idx])

        if self.transform:
            molecule = self.transform(molecule)

        return molecule
    
    def _download(self):
        pass

    def _process(self):
        pass
    
    def get(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        molecule = Molecule(self.X[idx],
                           self.y[idx])

        if self.transform:
            molecule = self.transform(molecule)

        return molecule