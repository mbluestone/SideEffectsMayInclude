import config
from utils_eval import evaluate_model

evaluate_model(data_dir=config.data_dir, # data directory
            model_type=config.model_type, # model type
            out_file="test_predictions.csv",
            model_path="trained_models/nlp_model.pt",
            graph_layers_sizes=config.graph_layers_sizes, 
            num_lstm_layers=config.num_lstm_layers, 
            nlp_embed_dim=config.nlp_embed_dim, 
            nlp_output_dim=config.nlp_output_dim,
            linear_layers_sizes=config.linear_layers_sizes, 
            dropout_rate=config.dropout_rate,
            batch_size=config.batch_size) 