import config
from utils_train import train_model

train_model(data_dir=config.data_dir,
            model_type=config.model_type,
            num_epochs=config.num_epochs,
            num_graph_layers=config.num_graph_layers,
            num_linear_layers=config.num_linear_layers,
            learning_rate=config.learning_rate,
            learning_rate_decay=config.learning_rate_decay,
            weight_decay=config.weight_decay, 
            batch_size=config.batch_size,
            log_csv=config.log_csv,
            log_file=config.log_file,
            pretrain_load_path=config.pretrain_load_path)