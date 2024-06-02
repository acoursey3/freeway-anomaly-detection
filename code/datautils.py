import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch_geometric.utils import to_networkx
import matplotlib.cm as cm
import networkx as nx
from tqdm import tqdm
import optuna

# Milemarkers present in the dataset
milemarkers = np.array([53.3, 53.6, 53.9, 54.1, 54.6, 55. , 55.3, 55.5, 56. , 56.3, 56.7,
       57.3, 57.7, 58.1, 58.3, 58.6, 58.8, 59. , 59.3, 59.7, 60. , 60.4,
       60.5, 61. , 61.5, 62.1, 62.5, 63.1, 63.6, 64. , 64.3, 64.5, 64.8,
       65. , 65.1, 65.6, 65.9, 66.3, 66.5, 66.7, 66.9, 67.3, 67.8, 68.2,
       68.5, 68.8, 69.3, 69.8, 70.1])

# Get all data from a specific lane and merge the together
def join_lane_data(day_no, lane_no):
    dfs = []
    features = ['occ', 'speed', 'volume']
    for feature in features:
        df = pd.read_csv(f"../data/day{day_no}/day_{day_no}_lane{lane_no}_{feature}.csv")
        dfs.append(df)
        
    lane_data = dfs[0]
    for df in dfs[1:]:
        lane_data = lane_data.merge(df)
        
    return lane_data[['time_index', 'milemarker', 'speed', 'occ', 'volume']]

# Get data from all lanes and combine them
def get_all_lanes(day_no, num_lanes):
    all_lanes = []
    
    for lane_no in range(num_lanes):
        curr_lane = join_lane_data(day_no, lane_no+1)
        all_lanes.append(curr_lane)
        
    return all_lanes    

# Get all data for a specific day
def get_day_data(day_no):
    all_lanes = get_all_lanes(day_no, 4)
    day_data = pd.concat(all_lanes).sort_values(by=['milemarker', 'time_index'])
    day_data['time_index'] = day_data['time_index'] - np.min(day_data['time_index'])
    return day_data

# Normalizes data based on maximum values
def normalize_data(training_data):
    normalized_data = training_data.copy()
    features = ['occ', 'speed', 'volume']
    max_vals = {
        'occ': 100,
        'speed': 80,
        'volume': 25
    }
    
    for feature in features:
        normalized_data[feature] /= max_vals[feature]
    
    return normalized_data

# Generates edge connections based on assumptions in paper
def generate_edges(milemarkers):
    num_nodes = len(milemarkers)*4
    edge_connections = []
    for i in range(num_nodes-4):
        lane_location = i % 4
        
        if lane_location == 0:
            # This is the left-most lane
            edge_connections.append([i, i+1]) # node to the right
            edge_connections.append([i, i+4])
            edge_connections.append([i, i+5])
            edge_connections.append([i, i+6])
            edge_connections.append([i, i+7])
        if lane_location == 1:
            # This is the second lane
            edge_connections.append([i, i+1]) # node to the right
            edge_connections.append([i, i+3])
            edge_connections.append([i, i+4])
            edge_connections.append([i, i+5])
            edge_connections.append([i, i+6])
        if lane_location == 2:
            # This is the third lane
            edge_connections.append([i, i+1]) # node to the right
            edge_connections.append([i, i+2])
            edge_connections.append([i, i+3])
            edge_connections.append([i, i+4])
            edge_connections.append([i, i+5])
        if lane_location == 3:
            # This is the right-most lane
            edge_connections.append([i, i+1])
            edge_connections.append([i, i+2])
            edge_connections.append([i, i+3])
            edge_connections.append([i, i+4])
           
    
    edge_connections.append([num_nodes-4-1, num_nodes-3-1])
    edge_connections.append([num_nodes-3-1, num_nodes-2-1])
    edge_connections.append([num_nodes-2-1, num_nodes-1-1])
    edge_connections.append([num_nodes-1-1, num_nodes-1])
    
    edge_connections = torch.tensor(edge_connections)
    # Since undirected, also add the reverse of all edges
    edge_connections = torch.cat([edge_connections, edge_connections.flip(dims=[1])], dim=0)
    
    return edge_connections.T
    
# Make PyTorch Geometric Data graph object
def make_graph(dataset):
    features = ['occ', 'speed', 'volume']
    x = dataset[features]
    x = np.array(x)
    
    undirected_edges = generate_edges(milemarkers)
    node_features = torch.tensor(x, dtype=torch.float32)
    
    graph = Data(x=node_features, edge_index=undirected_edges)
    return graph

# Make a time series of graphs for a day
def make_graph_sequence(day, times):
    full_day = get_day_data(day)

    graph_sequence = []
    for time in times:
        curr_time = full_day[full_day['time_index']==time]
        graph_sequence.append(make_graph(normalize_data(curr_time)))

    return graph_sequence

# Plot a graph
def plot_traffic_graph(graph):
    undirected_networkx_graph = to_networkx(graph, to_undirected=True)

    intensity_values = graph.x[:, 1].numpy()

    # Define a colormap (viridis in this case)
    cmap = cm.RdYlGn

    node_colors = [cmap((value)) for value in intensity_values]
    
    rows = len(graph.x) // 4  # Calculate the number of rows in the grid
    cols = 4

    # Calculate positions for each node in the grid
    pos = {i: (i % cols, -i // cols) for i in range(len(graph.x))}
    labels = {i: i for i in range(len(graph.x))}  # Node labels

    fig = plt.figure(figsize=(7,15))
    ax = plt.subplot(111)
    # Draw nodes and edges
    nx.draw(undirected_networkx_graph, pos=pos, ax=ax, with_labels=True, labels=labels,
            node_color=node_colors, font_color='black')
    plt.title("Traffic Speed", fontsize=32)
    plt.show()

# Turn a time series of graphs into a list of sequences of graphs of a certain time window (timesteps)
def temporalize_sequence(graph_sequence, timesteps=20, verbose=False):
    temporal_graphs = []

    for i in tqdm(range(timesteps-1, len(graph_sequence)), disable=not verbose):
        sliding_window = []
        for j in range(timesteps):
            sliding_window.append(graph_sequence[i-timesteps+1+j])
        
        temporal_graphs.append(sliding_window)

    return temporal_graphs

# Get data for a morning
def get_morning_data(day_no, timesteps):
    traffic_sequence = make_graph_sequence(day_no, list(range(120*12)))
    morning_data = temporalize_sequence(graph_sequence=traffic_sequence, timesteps=timesteps)

    return morning_data

# Get morning data for December 8
def get_dec8_data(timesteps):
    traffic_sequence = make_graph_sequence(0, list(range(120*10+20)))
    morning_data = temporalize_sequence(graph_sequence=traffic_sequence, timesteps=timesteps)

    return morning_data

def get_full_data():
    valid_days = [10, 11, 15, 16, 25] # chosen because they contain accidents of interest we want to detect (note index 10 is Oct 11 and so on)
    data = pd.read_csv('../data/data_with_label.csv')
    melted = pd.melt(data, id_vars=['day', 'milemarker', 'human_label', 'crash_record', 'unix_time'], value_vars=['lane1_speed', 'lane2_speed', 'lane3_speed', 'lane4_speed'], value_name='speed').sort_values(['unix_time', 'milemarker']).drop('variable', axis=1)
    melted2 = pd.melt(data, id_vars=['day', 'milemarker', 'human_label', 'crash_record', 'unix_time'], value_vars=['lane1_occ', 'lane2_occ', 'lane3_occ', 'lane4_occ'], value_name='occ').sort_values(['unix_time', 'milemarker']).drop('variable', axis=1)
    melted3 = pd.melt(data, id_vars=['day', 'milemarker', 'human_label', 'crash_record', 'unix_time'], value_vars=['lane1_volume', 'lane2_volume', 'lane3_volume', 'lane4_volume'], value_name='volume').sort_values(['unix_time', 'milemarker']).drop('variable', axis=1)
    melted['occ'] = melted2['occ']
    melted['volume'] = melted3['volume']

    melted = melted[melted['day'] != 17]

    train_data = melted[(melted['day'] != valid_days[0]) & (melted['day'] != valid_days[1]) & (melted['day'] != valid_days[2]) & (melted['day'] != valid_days[3]) & (melted['day'] != valid_days[4])]
    test_data = melted[(melted['day'] == valid_days[0]) | (melted['day'] == valid_days[1]) | (melted['day'] == valid_days[2]) | (melted['day'] == valid_days[3]) | (melted['day'] == valid_days[4])]
    return train_data, test_data, melted

def label_anomalies(data, include_manual=True):
    if include_manual:
        human_label_times = np.unique(data[data['human_label']==1]['unix_time'])
        for human_label_time in human_label_times:
            data.loc[(data['unix_time'] - human_label_time <= 7200) & (data['unix_time'] - human_label_time >= 0), 'anomaly'] = 1

    crash_label_times = np.unique(data[data['crash_record']==1]['unix_time'])
    for crash_label_time in crash_label_times:
        data.loc[(data['unix_time'] - crash_label_time <= 7200) & (data['unix_time'] - crash_label_time >= -1800), 'anomaly'] = 1

    data.fillna(0, inplace=True)

    return data

def get_rstae_sequence(day_no, timesteps, is_morning=True):
    sequence = []
    relational_edges, relations = generate_relational_edges(milemarkers=list(range(49)), timesteps=timesteps)
    static_edges = generate_edges(milemarkers=list(range(49)))
    day_data = get_day_data(day_no=day_no)
    day_normalized = normalize_data(day_data)

    if is_morning:
        day_normalized = day_normalized[day_normalized['time_index'] < 120*12]

    num_times = len(day_normalized['time_index'].unique())
    for t in range(timesteps, num_times): # skip first 'timesteps'
        data_t = []
        for i in range(timesteps-1, -1, -1):
            data_t.append(day_normalized[day_normalized['time_index']==t-i]) # assumes time indices come sequentially, with full data it may not

        combined = pd.concat(data_t)
        node_data = combined[['occ', 'speed', 'volume']].to_numpy()
        pyg_data = Data(x=torch.tensor(node_data, dtype=torch.float32), edge_index=relational_edges, edge_attr=torch.tensor(relations, dtype=torch.long)) # when should we put this on the GPU?
        
        curr_data = data_t[-1]
        curr_graph = Data(x=torch.tensor(curr_data[['occ', 'speed', 'volume']].to_numpy(), dtype=torch.float32), edge_index=static_edges)
        sequence.append([pyg_data, curr_graph])

    return sequence

def get_temporal_sequence(day_no, timesteps, is_morning=True):
    sequence = []
    day_data = get_day_data(day_no=day_no)
    day_normalized = normalize_data(day_data)

    if is_morning:
        day_normalized = day_normalized[day_normalized['time_index'] < 120*12]

    num_times = len(day_normalized['time_index'].unique())
    for t in range(timesteps, num_times): # skip first 'timesteps'
        data_t = []
        for i in range(timesteps-1, -1, -1):
            data_t.append(day_normalized[day_normalized['time_index']==t-i][['occ', 'speed', 'volume']]) # assumes time indices come sequentially, with full data it may not

        combined = np.array(data_t)
        combined = np.swapaxes(combined, 0, 1)
        combined = torch.tensor(combined, dtype=torch.float32)
        
        curr_data = combined[:,-1,:]
        sequence.append([combined, curr_data])

    return sequence

def get_gcnae_sequence(day_no, timesteps, is_morning=True):
    sequence = []
    static_edges = generate_edges(milemarkers=list(range(49)))
    day_data = get_day_data(day_no=day_no)
    day_normalized = normalize_data(day_data)

    if is_morning:
        day_normalized = day_normalized[day_normalized['time_index'] < 120*12]

    num_times = len(day_normalized['time_index'].unique())
    for t in range(timesteps, num_times): # skip first 'timesteps'
        curr_data = day_normalized[day_normalized['time_index'] == t]
        curr_graph = Data(x=torch.tensor(curr_data[['occ', 'speed', 'volume']].to_numpy(), dtype=torch.float32), edge_index=static_edges)
        sequence.append(curr_graph)

    return sequence

def generate_naive_edges(milemarkers):
    # Define the dimensions of the grid
    num_rows = len(milemarkers)
    num_cols = 4

    # Initialize an empty list to store edges
    edges = []

    # Generate edges for horizontal connections
    for i in range(num_rows):
        for j in range(num_cols - 1):
            node1 = i * num_cols + j
            node2 = i * num_cols + j + 1
            edges.append((node1, node2))

    # Generate edges for vertical connections
    for i in range(num_rows - 1):
        for j in range(num_cols):
            node1 = i * num_cols + j
            node2 = (i + 1) * num_cols + j
            edges.append((node1, node2))

    # Convert the list of edges to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Convert to undirected edges (optional)
    edge_index = to_undirected(edge_index)

    return edge_index


def generate_relational_edges(milemarkers, timesteps):
    # Definitions
    
    # Relation 0: Same time (CT), same milemarker (CM)
    # Relation 1: Same time, ahead milemarker (AM)
    # Relation 2: Same time, behind milemarker (BM)
    # Relation 3: Previous time (PT), same milemarker
    # Relation 4: Previous time, ahead milemarker
    # Relation 5: Previous time, behind milemarker
    relationships = {
        'CTCM': 0,
        'CTAM': 1,
        'CTBM': 2,
        'PTCM': 3,
        'PTAM': 4,
        'PTBM': 5
    }
    
    # Assume they are sequentially ordered, then we can add 196 to get the one in the future
    num_nodes = len(milemarkers)*4
    edge_connections = []
    relations = []
    for t in range(timesteps):
        time_offset = t*196
        for i in range(num_nodes-4):
            lane_location = i % 4
            
            curr_node = i+time_offset
            right_node = curr_node+1 # rewrite using this logic, much cleaner. Also get the ones for next time
            
            downl_node = curr_node+3
            downll_node = curr_node+2
            downlll_node = curr_node+1
            
            down_node = curr_node+4
            downr_node = curr_node+5 
            downrr_node = curr_node+6
            downrrr_node = curr_node+7
            
            if lane_location == 0:
                # This is the left-most lane
                edge_connections.append([curr_node, right_node]) # node to the right
                relations.append(relationships['CTCM'])
                edge_connections.append([right_node, curr_node]) # node to the right
                relations.append(relationships['CTCM'])
                
                edge_connections.append([curr_node, down_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([down_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downr_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downr_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downrr_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downrr_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downrrr_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downrrr_node, curr_node])
                relations.append(relationships['CTAM'])
                
                # Also need to connect similarly across time
            if lane_location == 1:
                # This is the second lane
                edge_connections.append([curr_node, right_node]) # node to the right
                relations.append(relationships['CTCM'])
                edge_connections.append([right_node, curr_node]) 
                relations.append(relationships['CTCM'])
                
                edge_connections.append([curr_node, downl_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downl_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, down_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([down_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downr_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downr_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downrr_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downrr_node, curr_node])
                relations.append(relationships['CTAM'])
                
            if lane_location == 2:
                # This is the third lane
                edge_connections.append([curr_node, right_node]) # node to the right
                relations.append(relationships['CTCM'])
                edge_connections.append([right_node, curr_node]) 
                relations.append(relationships['CTCM'])
                
                edge_connections.append([curr_node, downll_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downll_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downl_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downl_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, down_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([down_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downr_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downr_node, curr_node])
                relations.append(relationships['CTAM'])
            if lane_location == 3:
                # This is the right-most lane
                edge_connections.append([curr_node, downlll_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downlll_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downll_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downll_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, downl_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([downl_node, curr_node])
                relations.append(relationships['CTAM'])
                
                edge_connections.append([curr_node, down_node])
                relations.append(relationships['CTBM'])
                edge_connections.append([down_node, curr_node])
                relations.append(relationships['CTAM'])
                
        # Handle the last 4 nodes
        edge_connections.append([num_nodes+time_offset-4-1, num_nodes+time_offset-3-1])
        relations.append(relationships['CTCM'])
        edge_connections.append([num_nodes+time_offset-3-1, num_nodes+time_offset-4-1])
        relations.append(relationships['CTCM'])
        
        edge_connections.append([num_nodes+time_offset-3-1, num_nodes+time_offset-2-1])
        relations.append(relationships['CTCM'])
        edge_connections.append([num_nodes+time_offset-2-1, num_nodes+time_offset-3-1])
        relations.append(relationships['CTCM'])
        
        edge_connections.append([num_nodes+time_offset-2-1, num_nodes+time_offset-1-1])
        relations.append(relationships['CTCM'])
        edge_connections.append([num_nodes+time_offset-1-1, num_nodes+time_offset-2-1])
        relations.append(relationships['CTCM'])
        
        edge_connections.append([num_nodes+time_offset-1-1, num_nodes+time_offset-1])
        relations.append(relationships['CTCM'])
        edge_connections.append([num_nodes+time_offset-1, num_nodes+time_offset-1-1])
        relations.append(relationships['CTCM'])
        
        # Temporal connections
        if t < timesteps - 1:
            for i in range(num_nodes-4):
                lane_location = i % 4
            
                curr_node = i+time_offset
                past_node = curr_node+num_nodes
                pright_node = past_node+1 # rewrite using this logic, much cleaner. Also get the ones for next time
                
                pdownl_node = past_node+3
                pdownll_node = past_node+2
                pdownlll_node = past_node+1
                
                pdown_node = past_node+4
                pdownr_node = past_node+5 
                pdownrr_node = past_node+6
                pdownrrr_node = past_node+7
                
                edge_connections.append([curr_node, past_node]) 
                relations.append(relationships['PTCM'])
                edge_connections.append([past_node, curr_node]) 
                relations.append(relationships['PTCM'])
                
                if lane_location == 0:
                    # This is the left-most lane
                    edge_connections.append([curr_node, pright_node]) # node to the right
                    relations.append(relationships['PTCM'])
                    edge_connections.append([pright_node, curr_node]) # node to the right
                    relations.append(relationships['PTCM'])
                    
                    edge_connections.append([curr_node, pdown_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdown_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownr_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownr_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownrr_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownrr_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownrrr_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownrrr_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    # Also need to connect similarly across time
                if lane_location == 1:
                    # This is the second lane
                    edge_connections.append([curr_node, pright_node]) # node to the right
                    relations.append(relationships['PTCM'])
                    edge_connections.append([pright_node, curr_node]) 
                    relations.append(relationships['PTCM'])
                    
                    edge_connections.append([curr_node, pdownl_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownl_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdown_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdown_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownr_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownr_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownrr_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownrr_node, curr_node])
                    relations.append(relationships['PTAM'])
                if lane_location == 2:
                    # This is the third lane
                    edge_connections.append([curr_node, pright_node]) # node to the right
                    relations.append(relationships['PTCM'])
                    edge_connections.append([pright_node, curr_node]) 
                    relations.append(relationships['PTCM'])
                    
                    edge_connections.append([curr_node, pdownll_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownll_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownl_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownl_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdown_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdown_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownr_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownr_node, curr_node])
                    relations.append(relationships['PTAM'])
                if lane_location == 3:
                    # This is the right-most lane
                    edge_connections.append([curr_node, pdownlll_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownlll_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownll_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownll_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdownl_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdownl_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
                    edge_connections.append([curr_node, pdown_node])
                    relations.append(relationships['PTBM'])
                    edge_connections.append([pdown_node, curr_node])
                    relations.append(relationships['PTAM'])
                    
            # Handle the last 4 nodes
            edge_connections.append([num_nodes+time_offset-4-1, num_nodes+time_offset+num_nodes-4-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-4-1, num_nodes+time_offset-4-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset-4-1, num_nodes+time_offset+num_nodes-3-1])
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-3-1, num_nodes+time_offset-4-1])
            relations.append(relationships['PTCM'])
            
            edge_connections.append([num_nodes+time_offset-3-1, num_nodes+time_offset+num_nodes-3-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-3-1, num_nodes+time_offset-3-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset-3-1, num_nodes+time_offset+num_nodes-2-1])
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-2-1, num_nodes+time_offset-3-1])
            relations.append(relationships['PTCM'])
            
            edge_connections.append([num_nodes+time_offset-2-1, num_nodes+time_offset+num_nodes-2-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-2-1, num_nodes+time_offset-2-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset-2-1, num_nodes+time_offset+num_nodes-1-1])
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-1-1, num_nodes+time_offset-2-1])
            relations.append(relationships['PTCM'])
            
            edge_connections.append([num_nodes+time_offset-1-1, num_nodes+time_offset+num_nodes-1-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-1-1, num_nodes+time_offset-1-1]) 
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset-1-1, num_nodes+time_offset+num_nodes-1])
            relations.append(relationships['PTCM'])
            edge_connections.append([num_nodes+time_offset+num_nodes-1, num_nodes+time_offset-1-1])
            relations.append(relationships['PTCM'])
        
    edge_connections = torch.tensor(edge_connections)
    
    return edge_connections.T, relations

def load_best_parameters(study_name):
    # Load the study from the specified directory
    study = optuna.study.load_study(
        study_name=study_name,  # Replace with your study name
        storage=f"sqlite:///studies/{study_name}.db"  # Replace with the path to your SQLite database
    )

    # Get the best trial
    best_trial = study.best_trial

    # Get the best parameters
    best_params = best_trial.params

    return best_params