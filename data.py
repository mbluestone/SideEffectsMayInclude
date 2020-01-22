from pysmiles import read_smiles
import networkx as nx
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class Molecule():
    def __init__(self, smiles: str):
        
        
        self.smiles = smiles

        # create graph from smiles
        self.graph = read_smiles(self.smiles)
        
        # create adjacency matrix from graph
        self.adjacency_matrix = nx.to_numpy_matrix(self.graph, weight='order')
        
        self.atom_dict = pd.read_csv('raw_data/atom_info.txt',sep=',').set_index('Symbol')['AtomicNumber'].to_dict()
        
    def __str__(self):
        pass
    
    def __len__(self):
        return len(self.graph.nodes)
    
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
        
        feature_vector = np.array([])
        feature_vector = np.append(feature_vector,int(self.atom_dict[atom['element']]))
        feature_vector = np.append(feature_vector,atom['charge'])
        if atom['aromatic']:
            feature_vector = np.append(feature_vector,1)
        else:
            feature_vector = np.append(feature_vector,0)
        feature_vector = np.append(feature_vector,atom['hcount'])
        
        return feature_vector
        
class MoleculeDataset(Dataset):
    """Molecules dataset."""

    def __init__(self, smiles_list, transform=None):
        """
        Args:
            smiles_list (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.smiles_list = smiles_list
        self.transform = transform

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        molecule = Molecule(self.smiles_list[idx])
        adjacency = molecule.adjacency_matrix
        features = molecule.extract_features()
        sample = {'adjacency': adjacency, 'features': features}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        adjacency, features = sample['adjacency'], sample['features']

        return {'adjacency': torch.from_numpy(adjacency),
                'features': torch.from_numpy(features)}