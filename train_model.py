import config
from utils_train import train_model

train_model(data_dir=config.data_dir, # data directory
            model_type=config.model_type, # model type
            # graph model parameters
            graph_layers_sizes=config.graph_layers_sizes,
            # nlp model parameters
            num_lstm_layers=config.num_lstm_layers,
            nlp_embed_dim=config.nlp_embed_dim,
            nlp_output_dim=config.nlp_output_dim,
            # fully connected linear model parameters
            linear_layers_sizes=config.linear_layers_sizes,
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