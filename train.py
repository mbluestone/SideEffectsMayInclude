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

#### DATA #####
#load
data_path = '~/github/MolNet/raw_data/sider_data/sider.csv'
sider_data = pd.read_csv(data_path)
smiles_list = sider_data.smiles.to_list()
dataset = MoleculeDataset(data_path, transform=AddSelfLoops())

# temporary
for i in range(sider_data.shape[0]):
    dataset[i].y = torch.tensor([sider_data.iloc[i,1:].to_list()],
                                 dtype=torch.float32)
dataloader = DataLoader(dataset,batch_size=5,shuffle=True)

train_path = '~/github/MolNet/raw_data/sider_data/train_sider.csv'
val_path = '~/github/MolNet/raw_data/sider_data/val_sider.csv'
test_path = '~/github/MolNet/raw_data/sider_data/test_sider.csv'

train_data = train_data.reset_index().drop('index',axis=1)
val_data = val_data.reset_index().drop('index',axis=1)
test_data = test_data.reset_index().drop('index',axis=1)

train_data.to_csv(train_path)
val_data.to_csv(val_path)
test_data.to_csv(test_path)

train_dataset = MoleculeDataset(train_path, transform=AddSelfLoops())
val_dataset = MoleculeDataset(val_path, transform=AddSelfLoops())
test_dataset = MoleculeDataset(test_path, transform=AddSelfLoops())

train_dataloader = DataLoader(train_dataset,batch_size=50,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=50,shuffle=True)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# intialize data loaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes


#### MODEL CLASS #####
class MolNet(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int):
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
        sum_vector = global_add_pool(x,batch = batch_vec)
        x = F.relu(sum_vector)
        x = F.dropout(x, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)

        return x

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def create_model(num_layers: int, num_classes: int,
                 pretrain: bool) -> torchvision.models.resnet.ResNet:
    """
    Instantiate the ResNet model.
    Args:
        num_layers: Number of layers to use in the ResNet model from [18, 34, 50, 101, 152].
        num_classes: Number of classes in the dataset.
        pretrain: Use pretrained ResNet weights.
    Returns:
        The instantiated ResNet model with the requested parameters.
    """
    assert num_layers in (
        18, 34, 50, 101, 152
    ), f"Invalid number of ResNet Layers. Must be one of [18, 34, 50, 101, 152] and not {num_layers}"
    model_constructor = getattr(torchvision.models, f"resnet{num_layers}")
    model = model_constructor(num_classes=num_classes)

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

def train_model():
    '''
    Function for training model
    
    Args:
        num_epochs
    '''
    
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_node_features = 4
    num_classes = train_dataset[0].y.size(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_node_features, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    criterion = BCEWithLogitsLoss()
    
    pos_weights = get_pos_weights()
    

def train_helper(model, num_epochs, dataloaders, pos_weights):
    '''
    Helper function for training model
    
    Args:
        model:
        num_epochs
    '''

    for epoch in range(num_epochs):

        print(f'Epoch {epoch}:')
        model.train()

        train_running_loss = 0.0
        train_running_accuracy = 0.0

        for inputs in train_dataloader:

            optimizer.zero_grad()
            with torch.set_grad_enabled(mode=True):
                out = model(inputs)
                train_loss = criterion(out, inputs.y)
                train_acc = ((torch.sigmoid(out)>0.5).numpy()==inputs.y.numpy()).mean()
                train_loss.backward()
                optimizer.step()

            train_running_loss += train_loss.item() * inputs.y.size(0)
            train_running_accuracy += train_acc * inputs.y.size(0)

        epoch_train_loss = train_running_loss/len(train_dataset)
        train_losses.append(epoch_train_loss)
        epoch_train_acc = train_running_accuracy/len(train_dataset)
        train_accuracies.append(epoch_train_acc)

        print(f'Train:\nLoss = {epoch_train_loss}\nAcc = {epoch_train_acc}')   

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

        print(f'Validation:\nLoss = {epoch_val_loss}\nAcc = {val_accuracies}')