data_dir='processed_data/sider/'

# model creation parameters
model_type='nlp'
pretrain_load_path=None

# graph model parameters
graph_layers_sizes=[15,20,36]

# nlp model parameters
num_lstm_layers=1
nlp_embed_dim=100
nlp_output_dim=50

# fully connect linear model parameters
linear_layers_sizes=[100,200,100]

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
