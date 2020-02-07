import sys
sys.path.append("../code")

from os.path import dirname
from os.path import join as path_join

import torch
import dill

from utils_eval import load_model
from utils_data import Molecule, process_smiles_for_nlp
 

def make_prediction(model_path,input_smiles):
    
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load TEXT field from 
    with open(path_join(dirname(model_path),"TEXT.Field"),"rb") as f:
         TEXT=dill.load(f)
    
    model, model_params_dict = load_model(model_path, device) 
    
    input_molecule = Molecule(input_smiles,[],atom_info_path='../raw_data/atom_info.txt')
    input_molecule.batch = torch.tensor([0])
    input_molecule.text = process_smiles_for_nlp(input_smiles,TEXT.vocab.stoi,200).view(1,-1)
    
    print('x',input_molecule.x.size())
    print('edge_index',input_molecule.edge_index.size())
    print('batch',input_molecule.batch.size())
    print('text',input_molecule.text.size())
    
    input_molecule.to(device)
    
    output = model(input_molecule)
    
    probs = torch.sigmoid(output).detach().cpu().numpy()
    predictions = (torch.sigmoid(output)>0.5).detach().cpu().numpy()
    
    return probs, predictions