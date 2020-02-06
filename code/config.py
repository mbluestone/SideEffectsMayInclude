# directory to find the data
data_dir='processed_data/sider/all_side_effects/'
label='blood pressure increased'

# model creation parameters
model_type='combo'
pretrain_load_path=None

# graph model parameters
graph_layers_sizes=[16,20]

# nlp model parameters
text_embed_dim=30
text_output_dim=20

# fully connect linear model parameters
linear_layers_sizes=[50,80]

# model training parameters
num_epochs=50
learning_rate=0.0001
learning_rate_decay=0.95 
weight_decay=1e-4 
dropout_rate=0.4
batch_size=64

# logging
log_file=None
log_csv = 'logs/combo_log.csv'
