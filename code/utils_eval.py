from utils_data import *


###########################################
#            MODEL EVALUATION             #
###########################################

def load_model(load_path: dict,
               device: torch.device) -> FullModel:
    """
    Instantiate the model from saved dictionary of pretrained model.
    Args:
        num_graph_layers: Number of layers to use in the model.
        num_classes: Number of classes in the dataset.
        pretrain_load_path: Use pretrained weights.
    Returns:
        The instantiated model with the requested parameters.
    """

    ckpt = torch.load(f=load_path)
    model_params_dict = ckpt["model_params_dict"]    
    model = FullModel(model_type=model_params_dict['model_type'], 
                      num_classes=model_params_dict['num_classes'], 
                      num_node_features=model_params_dict['num_node_features'], 
                      graph_layers_sizes=model_params_dict['graph_layers_sizes'],  
                      num_lstm_layers=model_params_dict['num_lstm_layers'], 
                      nlp_embed_dim=model_params_dict['nlp_embed_dim'], 
                      nlp_output_dim=model_params_dict['nlp_output_dim'], 
                      linear_layers_sizes=model_params_dict['linear_layers_sizes'], 
                      dropout_rate=model_params_dict['dropout_rate'],
                      vocab_size=model_params_dict['vocab_size'])

    model.load_state_dict(state_dict=ckpt["model_state_dict"], 
                          map_location=device)
        
    # transfer model to cpu or gpu
    model = model.to(device=device)
        
    return model, model_params_dict

def evaluate_model(model_path, 
                   test_data_dir, 
                   out_file):
    
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load the trained model
    model, model_params_dict = load_model(load_path=model_path,
                                          device=device)
    print(f"model loaded from {model_path}")
    
    # load data objects
    dataloaders,dataset_sizes,pos_weight,labels,num_node_features,vocab_size = load_data_for_model(test_data_dir, 
                                                                                                   device, 
                                                                                                   model_params_dict['model_type'], 
                                                                                                   model_params_dict['batch_size'], 
                                                                                                   ngram=2, 
                                                                                                   training=False)
    
    # Testing
    model.eval()

    # initialize running metrics
    running_accuracy = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    running_roc_auc = 0.0

    all_labels = np.array([])
    all_probs = np.array([])
    all_predictions = np.array([])

    # loop through batched validation data
    for inputs in dataloaders['test']:

        # pull out batch labels
        batch_labels = inputs.y.cpu().numpy()
        print(batch_labels)

        # send to device
        inputs.y = inputs.y.to(device)
        inputs.x = inputs.x.to(device)
        if 'edge_index' in dir(inputs):
            inputs.edge_index = inputs.edge_index.to(device)
            inputs.batch = inputs.batch.to(device)

        with torch.set_grad_enabled(mode=False):

            # make predicitions
            out = model(inputs)

            batch_probs = torch.sigmoid(out).detach().cpu().numpy()
            batch_predictions = (torch.sigmoid(out)>0.5).detach().cpu().numpy()

            # calculate performance metrics
            batch_accuracy = 1-hamming_loss(batch_labels,batch_predictions)
            batch_precision = precision_score(batch_labels,batch_predictions,
                                            average='micro',zero_division=0)
            batch_recall = recall_score(batch_labels,batch_predictions,
                                      average='micro',zero_division=0)
            batch_f1 = f1_score(batch_labels,batch_predictions,
                              average='micro',zero_division=0)
            batch_roc_auc = roc_auc_score(batch_labels,batch_probs,
                                        average='micro')

        # update running metrics
        running_accuracy += batch_accuracy * inputs.y.size(0)
        running_precision += batch_precision * inputs.y.size(0)
        running_recall += batch_recall * inputs.y.size(0)
        running_f1 += batch_f1 * inputs.y.size(0)
        running_roc_auc += batch_roc_auc * inputs.y.size(0)

        if all_labels.size == 0:
            all_labels = batch_labels
            all_probs = batch_probs
            all_predictions = batch_predictions
        else:
            all_labels = np.vstack((all_labels,batch_labels))
            all_probs = np.vstack((all_probs,batch_probs))
            all_predictions = np.vstack((all_predictions,batch_predictions))

    # calculate validation metrics for the epoch
    accuracy = np.round(running_accuracy/dataset_sizes['test'],decimals=4)
    precision = np.round(running_precision/dataset_sizes['test'],decimals=4)
    recall = np.round(running_recall/dataset_sizes['test'],decimals=4)
    f1 = np.round(running_f1/dataset_sizes['test'],decimals=4)
    roc_auc = np.round(running_roc_auc/dataset_sizes['test'],decimals=4)

    print(f'Test:\n'
          f'Accuracy = {accuracy}, '
          f'Precision = {precision}, '
          f'Recall = {recall}, '
          f'F1 = {f1}, '
          f'ROC_AUC = {roc_auc}\n') 

    # print confusion matrices
    for i,label in enumerate(labels):
        print('\n',label,':\n')
        print(confusion_matrix(all_labels[:,i],all_predictions[:,i]))
        
    pd.DataFrame(all_probs,columns=labels).to_csv(out_file,index=False)
    
###########################################
#            MODEL PREDICTIONS            #
###########################################
    
def get_prediction(model,input_smiles):
    '''
    Get model predictions for one input 
    '''
    
    data = smiles_to_model_data(input_smiles)
    output = model(data)
    
    probs = torch.sigmoid(output).detach().cpu().numpy()
    predictions = (torch.sigmoid(output)>0.5).detach().cpu().numpy()
    
    return probs, predictions
    