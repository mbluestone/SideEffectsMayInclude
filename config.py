data_dir='processed_data/sider/'

# model creation parameters
model_type='graph'
num_graph_layers=2
num_linear_layers=3
pretrain_load_path=None

# model training parameters
num_epochs=10
learning_rate=0.01
learning_rate_decay=0.85 
weight_decay=1e-4 
dropout_rate=0.5
batch_size=30

# logging
log_file=None
log_csv = 'logs/test_log.csv'
