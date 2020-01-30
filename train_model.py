import config
from utils_train import train_model

train_model(data_dir=config.data_dir, # data directory
            model_type=config.model_type, # model type
            # graph model parameters
            num_graph_layers=config.num_graph_layers
            num_graph_linear_layers=config.num_graph_linear_layers
            num_graph_linear_nodes=config.num_graph_linear_nodes
            # nlp model parameters
            num_nlp_linear_layers=config.num_nlp_linear_layers
            num_nlp_linear_nodes=config.num_nlp_linear_nodes
            embed_size=config.embed_size
            # model training parameters
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            learning_rate_decay=config.learning_rate_decay,
            weight_decay=config.weight_decay, 
            dropout_rate=config.dropout_rate,
            batch_size=config.batch_size,
            # log files
            log_csv=config.log_csv,
            log_file=config.log_file,
            # if loading pretrained model
            pretrain_load_path=config.pretrain_load_path) 