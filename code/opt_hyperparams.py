import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna
import os

from models import SpatioTemporalAutoencoder
from parameters import GATAEParameters, RSTAEParameters, GraphAEParameters, MLPAEParameters, TrainingParameters
from datautils import get_rstae_sequence, get_gcnae_sequence, get_temporal_sequence
from training import train_gatae, train_rstae, train_gcnae, train_mlpae, test_rstae, test_gcnae, test_mlpae

model_type = None

def choose_parameters(trial):
    if model_type == "rstae":
        parameters = RSTAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128, 256]),
            gcn_hidden_dim=trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
            dropout=trial.suggest_float('dropout', 0, 0.5),
            num_gcn=trial.suggest_int('num_gcn', 1, 5)
        )
    elif model_type == "gcn":
        parameters = GraphAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128, 256]),
            gcn_hidden_dim=trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
            dropout=trial.suggest_float('dropout', 0, 0.5),
            num_gcn=trial.suggest_int('num_gcn', 1, 5)
        )
    elif model_type == "gat":
        parameters = GATAEParameters(
            num_features=3,
            latent_dim=trial.suggest_categorical('latent_dim', [32, 64, 128, 256]),
            gcn_hidden_dim=trial.suggest_categorical('gcn_hidden_dim', [16, 32, 64, 128, 256]),
            dropout=trial.suggest_float('dropout', 0, 0.5),
            num_layers=trial.suggest_int('num_layers', 1, 5),
            num_heads=trial.suggest_int('num_heads', 1, 5)
        )
    elif model_type == "mlp":
        parameters = MLPAEParameters(
            num_features=3,
            latent_dim=2,
            hidden_dim=trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])
        )
    else:
        raise NotImplementedError("Please one of the allowed model types.")
    
    return parameters

def train_model(hyperparams, training_params, training_data, mse_weights, verbose=False):
    if model_type == "rstae":
        model, losses = train_rstae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "gcn":
        model, losses = train_gcnae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "gat":
        model, losses = train_gatae(hyperparams, training_params, training_data, mse_weights, verbose)
    elif model_type == "mlp":
        model, losses = train_mlpae(hyperparams, training_params, training_data, mse_weights, verbose)
    
    return model, losses

def get_data(day_no, timesteps):
    if model_type == "rstae" or model_type == "gat":
        data = get_rstae_sequence(day_no, timesteps, is_morning=True)
    elif model_type == "gcn":
        data = get_gcnae_sequence(day_no, timesteps, is_morning=True)
    elif model_type == "mlp":
        data = get_temporal_sequence(day_no, timesteps=1, is_morning=True)
        
    return data

def validate_model(valid_data, mse_weights, model):
    if model_type == "rstae" or model_type == "gat":
        errors, _, _ = test_rstae(valid_data, mse_weights, model)
    elif model_type == "gcn":
        errors, _, _ = test_gcnae(valid_data, mse_weights, model)
    elif model_type == "mlp":
        errors, _, _ = test_mlpae(valid_data, mse_weights, model)
    
    return errors

def objective(trial):
    best_mse = trial.study.user_attrs.get('best_mse', float('inf'))
    train_params = TrainingParameters(
        learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
        batch_size=1,
        timesteps=trial.suggest_int('timesteps', 2, 10) if not model_type=="mlp" else 0,
        n_epochs=trial.suggest_int('epochs', 1, 10),
    )

    # Crash-free morning as training
    train_data = get_data(1, train_params.timesteps)

    params = choose_parameters(trial)

    mse_weights = [1,1,1] # loss function weights
    model, losses = train_model(hyperparams=params, training_params=train_params, training_data=train_data, mse_weights=mse_weights, verbose=False)

    # Using another crash-free morning as validation
    valid_data = get_data(6, train_params.timesteps) 

    errors = validate_model(valid_data, mse_weights, model)
    
    curr_mse = float(np.mean(errors))
    # if better than best mse, call save_model function
    # if curr_mse < best_mse:
    #     trial.study.set_user_attr('best_mse', curr_mse)
    #     save_model(model, f'opt_{model_type}_{trial.number}')

    return curr_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-m', '--model',
        choices=['rstae', 'gcn', 'mlp', 'gat'],
        required=True,
        help='Choose a model: rstae, gcn, mlp, gat'
    )
    
    args = parser.parse_args()
    study_name = args.model # what to call the study
    # study_name = study_name + "_v2"
    model_type = args.model # used in optuna optimization to choose relevant functions
    
    storage_subdirectory = 'studies'
    storage_url = f'sqlite:///{os.path.join(storage_subdirectory, study_name)}.db'

    study = optuna.create_study(direction='minimize', study_name=study_name, storage=storage_url, load_if_exists=True)
    study.optimize(objective, n_trials=100, n_jobs=8)