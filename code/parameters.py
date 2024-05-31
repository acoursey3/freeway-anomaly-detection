from dataclasses import dataclass

# Parameters for the Spatiotemporal Autoencoder
@dataclass
class STAEParameters:
    num_features: int
    latent_dim: int
    gcn_hidden_dim: int
    lstm_hidden_dim: int
    lstm_num_layers: int
    dropout: int
    
# Parameters for the training process
@dataclass
class TrainingParameters:
    learning_rate: float
    batch_size: int
    timesteps: int
    n_epochs: int

# Parameters for the GAT Spatiotemporal Autoencoder
@dataclass
class GATSTAEParameters:
    num_features: int
    latent_dim: int
    gcn_hidden_dim: int
    lstm_hidden_dim: int
    lstm_num_layers: int
    dropout: int
    gat_heads: int

@dataclass
class RSTAEParameters:
    num_features: int
    latent_dim: int
    gcn_hidden_dim: int
    dropout: int
    num_gcn: int
    
@dataclass
class GraphAEParameters:
    num_features: int
    latent_dim: int
    gcn_hidden_dim: int
    dropout: int
    num_gcn: int
    
@dataclass
class TransformerAEParameters:
    num_features: int
    latent_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    sequence_length: int
    
@dataclass
class GATAEParameters:
    num_features: int
    latent_dim: int
    gcn_hidden_dim: int
    dropout: int
    num_heads: int
    num_layers: int