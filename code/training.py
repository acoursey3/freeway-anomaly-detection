import torch
import numpy as np
import matplotlib.pyplot as plt

from models import GATAE, RelationalSTAE, SpatioTemporalAutoencoder, GATSpatioTemporalAutoencoder, GraphAE, TransformerAutoencoder
from parameters import GATAEParameters, RSTAEParameters, STAEParameters, TrainingParameters, GATSTAEParameters, GraphAEParameters, TransformerAEParameters
from datautils import get_morning_data, milemarkers
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import Data as PyGData
import pandas as pd

import random

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, input, target):
        # Calculate squared errors
        errors = (input - target) ** 2

        # Apply weights to squared errors
        weighted_errors = errors * self.weights

        # Calculate the mean over all dimensions
        loss = torch.mean(weighted_errors)

        return loss
    
def save_model(model, name):
    torch.save(model.state_dict(), f'./saved_models/{name}.pth')

def load_model(modelclass, parameters, name):
    model = modelclass(parameters)
    checkpoint = torch.load(f'./saved_models/{name}.pth')
    model.load_state_dict(checkpoint)

    return model

def train_stae(staeparams: STAEParameters, trainingparams: TrainingParameters, training_data: PyGData, mse_weights: list = [1,1,1], verbose=False, full_data=False):
    ae = SpatioTemporalAutoencoder(staeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            if full_data:
                graph_sequence = graph_sequence[0]
            optimizer.zero_grad()
            xhat = ae(graph_sequence) # encode and decode the sequence

            loss = weighted_mse(xhat, graph_sequence[-1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_rstae(rstaeparams: RSTAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = RelationalSTAE(rstaeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for rgraph in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(rgraph[0]) # encode and decode the sequence

            loss = weighted_mse(xhat, rgraph[1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_gatae(gataeparams: GATAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = GATAE(gataeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for rgraph in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(rgraph[0]) # encode and decode the sequence

            loss = weighted_mse(xhat, rgraph[1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_transformerae(params: TransformerAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = TransformerAutoencoder(params)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for sequence, current in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(sequence) # encode and decode the sequence

            loss = weighted_mse(xhat, current)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_gcnae(gcnaeparams: GraphAEParameters, trainingparams: TrainingParameters, training_data, mse_weights: list = [1,1,1], verbose=False):
    ae = GraphAE(gcnaeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for rgraph in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(rgraph) # encode and decode the sequence

            loss = weighted_mse(xhat, rgraph.x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

def train_gatstae(staeparams: GATSTAEParameters, trainingparams: TrainingParameters, training_data: PyGData, mse_weights: list = [1,1,1], verbose=False):
    ae = GATSpatioTemporalAutoencoder(staeparams)
    optimizer = torch.optim.Adam(params=ae.parameters(), lr=trainingparams.learning_rate)
    weighted_mse = WeightedMSELoss(weights = mse_weights)

    losses = []

    for epoch_num in tqdm(range(trainingparams.n_epochs), disable=not verbose):
        shuffled_sequence = random.sample(training_data, len(training_data))
        for graph_sequence in shuffled_sequence:
            optimizer.zero_grad()
            xhat = ae(graph_sequence)

            loss = weighted_mse(xhat, graph_sequence[-1].x)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if verbose:
            print(f'Epoch number {epoch_num} last 100 loss {np.mean(losses[-100:])}')

    return ae, losses

# Computes weighted reconstruction error
def compute_weighted_error(error, weights):
    error *= np.array(weights) 
    error = np.mean(error, axis=1)

    return error

# Determines anomaly threshold
def compute_anomaly_threshold(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for graph_sequence in tqdm(training_data):
        xhat = model(graph_sequence[1])
        error = (xhat - graph_sequence[-1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
def compute_anomaly_threshold_rstae(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for graph_sequence in tqdm(training_data):
        xhat = model(graph_sequence[0])
        error = (xhat - graph_sequence[1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
def compute_anomaly_threshold_transformerae(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for sequence, current in tqdm(training_data):
        xhat = model(sequence)
        error = (xhat - current) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
def compute_anomaly_threshold_gcnae(training_data, model, weights, method='max'):
    model.eval()
    errors = []

    for graph_sequence in tqdm(training_data):
        xhat = model(graph_sequence)
        error = (xhat - graph_sequence.x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        errors.append(weighted)

    errors = np.array(errors)
    if method == 'max': # max reconstruction error on training data
        return np.max(errors, axis=0)
    elif method == 'mean': # mean + 3 * std reconstruction error
        return np.mean(errors, axis=0) + 3*np.std(errors, axis=0)
    else:
        raise NotImplementedError()
    
# Determine which nodes are anomalies based on threshold
def threshold_anomalies(thresh, errors):
    anomalies = []
    
    for error_t in errors:
        anomalies.append((error_t > thresh).astype(int))

    return np.array(anomalies)
    
# Evaluate a model on the test set
def test_model(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for graph_sequence in tqdm(test_sequence, disable=not verbose):
        xhat = model(graph_sequence)
        error = (xhat - graph_sequence[-1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(graph_sequence[-1].x[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

def test_rstae(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for graph_sequence in tqdm(test_sequence, disable=not verbose):
        xhat = model(graph_sequence[0])
        error = (xhat - graph_sequence[1].x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(graph_sequence[1].x[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

def test_transformerae(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for sequence, current in tqdm(test_sequence, disable=not verbose):
        xhat = model(sequence)
        error = (xhat - current) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(current[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

def test_gcnae(test_sequence, weights, model, verbose=False):
    model.eval()
    errors = []
    recons_speeds = []
    speeds = []

    for graph_sequence in tqdm(test_sequence, disable=not verbose):
        xhat = model(graph_sequence)
        error = (xhat - graph_sequence.x) ** 2
        error = error.detach().numpy()
        weighted = compute_weighted_error(error, weights)
        recons_speeds.append(xhat.detach().numpy()[:,1])
        speeds.append(graph_sequence.x[:,1].detach().numpy())
        errors.append(weighted)

    errors = np.array(errors)
    recons_speeds = np.array(recons_speeds)
    speeds = np.array(speeds)

    return errors, recons_speeds, speeds

# Create a df to save results
def build_result_df(anomalies, lanes, mms, times, speeds, recons_speeds):
    results = pd.DataFrame({
        'Anomaly': anomalies,
        'Lane': lanes,
        'Milemarker': mms,
        'Time Index': times,
        'Speed': speeds,
        'Reconstructed Speed': recons_speeds
    })

    return results

# Convert index to time
def time_convert(time_indices: np.ndarray):
    return time_indices * 30

# Convert node indices to a list of the lane numbers for those nodes
def node_lane_convert(node_indices: np.ndarray):
    lane_num = node_indices % 4 + 1
    return lane_num

# Convert node indices to their corresponding milemarkers
def node_milemarker_convert(node_indices: np.ndarray, milemarkers: np.ndarray):
    milemarker_num = milemarkers[node_indices // 4]
    return milemarker_num

# Put all results into the df
def fill_result_df(anomalies, speeds, recons_speeds, timesteps):
    mms = node_milemarker_convert(np.array(range(anomalies.shape[1])), milemarkers)
    mms = np.tile(mms, anomalies.shape[0])

    lanes = node_lane_convert(np.array(range(anomalies.shape[1])))
    lanes = np.tile(lanes, anomalies.shape[0])

    times = np.array(range(anomalies.shape[0])) + timesteps - 1
    times = times.repeat(anomalies.shape[1])

    df = build_result_df(anomalies.flatten(), lanes, mms, times, speeds.flatten(), recons_speeds.flatten())

    return df