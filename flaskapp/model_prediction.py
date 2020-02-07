import sys
sys.path.append("../code")

from os.path import dirname
from os.path import join as path_join

import torch
import dill
import pandas as pd

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
    input_molecule.batch = torch.LongTensor([0]*len(input_molecule.x))
    input_molecule.text = process_smiles_for_nlp(input_smiles,TEXT.vocab.stoi,200).view(1,-1)
    
    print('x',input_molecule.x.size())
    print('edge_index',input_molecule.edge_index.size())
    print('batch',input_molecule.batch.size())
    print('text',input_molecule.text.size())
    
    input_molecule = input_molecule.to(device)
    
    output = model(input_molecule)
    
    probs = torch.sigmoid(output).detach().cpu().numpy()
    prediction_mask = (torch.sigmoid(output)>0.5).detach().cpu().numpy()[0]
    labels = pd.read_csv('../processed_data/sider/common_sider.csv').columns.tolist()[1:]
    predictions = {labels[i]:probs[0][i] for i in range(len(labels)) if prediction_mask[i]}
    
    return predictions, probs