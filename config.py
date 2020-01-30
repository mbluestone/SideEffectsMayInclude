data_dir='processed_data/sider/'

# model creation parameters
model_type='graph'
pretrain_load_path=None

# graph
if model_type=='graph'
    num_graph_layers=2
    num_linear_layers=3
    num_linear_nodes=100

# nlp
if model_type=='nlp':
    num_linear_layers=3
    num_linear_nodes=100
    embed_size=100

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
