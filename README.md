# SideEffectsMayInclude

SideEffectsMayInclude is package to train and deploy a deep neural network model in PyTorch to predict drug side effects from their molecular structure. The model employs Graph Convolutional Network layers and a Bidirectional LSTM to separately extract signal from the molecular structure in graph form and in a SMILES string, respectively.

### Installation

To use this tool, you must first clone or download the GitHub repo wherever you will training/serving the model.

Then, build the Docker image:

`docker build`

### Data

Input data needs to be in CSV file with the rows representing each drug. The first column should be the SMILES string for each drug, while the rest of the columns should be a binary label for each side effect in the data set. An example of data in the correct format can be found in the `preprocessed_data/` folder.

The `split_data` function in `utils_data.py` can be used to split an imbalanced multilabel dataset in an iterative fashion to preserve class distribution across training, validation, and testing sets. The split datasets are saved as `train.csv`, `val.csv`, and `test.csv`.

### Usage

Train the model on the training data with:

`python code/train_model.py`

Evaluate model predictions on the testing data with:

`python code/test_model.py`

To start the Flask app, run `./run.py` from inside `flaskapp/`.
