data_dir='~/github/MolNet/processed_data/sider/'

# model creation parameters
model_type='graph'
num_graph_layers=2
num_linear_layers=3
pretrain_load_path=None

# model training parameters
num_epochs=100
learning_rate=0.001
learning_rate_decay=0.85 
weight_decay=1e-4 
batch_size=30
log_csv = '/Users/mbluestone/github/MolNet/logs/test_log.csv',
