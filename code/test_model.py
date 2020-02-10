import config
from utils_eval import evaluate_model

evaluate_model(data_dir=config.data_dir,
               out_file=config.test_out_file,
               model_path=config.test_model_path) 

