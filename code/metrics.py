import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from training import threshold_anomalies
from sklearn.metrics import roc_curve, auc 

# Anomaly labels are generated under the following assumptions
# If an event has been manually labeled, there is no delay in reporting.
# If a crash has been reported, there is at most a 15 minute delay in reporting, so any prediction 15 minutes before is correct.
# The end of an event cannot be accurately determined. Therefore, data 15 minutes after an event is still anomalous. 
# The next 1:45 cannot be accurately determined to be nominal or anomalous.
# These assumptions allow us to get a grasp of our detection accuracy, but clearly are very conservative and lead to true positive rate being less meaningful.
def generate_anomaly_labels(test_data, kept_indices):
    unix_times = np.unique(test_data['unix_time'])
    test_data = test_data[test_data['unix_time'].isin(unix_times[kept_indices])]
    human_label_times = np.unique(test_data[test_data['human_label']==1]['unix_time'])
    for human_label_time in human_label_times:
        test_data.loc[(test_data['unix_time'] - human_label_time <= 1800) & (test_data['unix_time'] - human_label_time >= 0), 'anomaly'] = 1

    crash_label_times = np.unique(test_data[test_data['crash_record']==1]['unix_time'])
    for crash_label_time in crash_label_times:
        test_data.loc[(test_data['unix_time'] - crash_label_time <= 900) & (test_data['unix_time'] - crash_label_time >= -900), 'anomaly'] = 1

    incident_times = np.unique(test_data[(test_data['human_label']==1) | (test_data['crash_record']==1)]['unix_time'])
    for incident_time in incident_times:
        test_data.loc[(test_data['unix_time'] - incident_time <= 6300) & (test_data['unix_time'] - incident_time >= 900), 'anomaly'] = -1
    
    test_data.fillna(0, inplace=True)

    return test_data['anomaly'].to_numpy()

def discrete_fp_delays(thresh, test_errors, anomaly_labels, crash_reported):
    thresholds = find_thresholds(thresh, test_errors, anomaly_labels)

    fprs = [1, 2.5, 5, 10, 20]
    new_thresholds = []
    for fpr in fprs:
        new_thresholds.append(find_percent(thresholds, fpr))

    anomaly_instances = []
    for t in new_thresholds:
        anomaly_instances.append(threshold_anomalies(thresh+t, test_errors))

    for i, fpr in enumerate(fprs):
        delay, found = crash_detection_delay(anomaly_instances[i], crash_reported)
        mu = np.mean(delay) / 2
        std = np.std(delay) / 2
        miss_percent = 1-(np.sum(found) / len(found))
        print(f'FPR {fpr}% gives mean delay of {mu} +/- {std} while missing {miss_percent}%.')

def find_percent(thresholds, percent):
    percent = percent / 100
    thresholds = np.array(thresholds)
    index_closest = thresholds.shape[0] - 1 - np.argmin(np.abs(thresholds[:,0][::-1] - percent))
    return thresholds[index_closest][1]

def find_delays(thresh, errors, anomaly_labels, crash_reported):
    results = []
    thresholds = np.array(find_thresholds(thresh, errors, anomaly_labels))
    all_fp_index = np.where(thresholds[:,0] == 1)[0][-1]
    no_fp_index = np.where(thresholds[:,0] == 0)[0][-1]
    val_range = np.linspace(0.01, 0.99, 98)
    
    anomaly_pred = threshold_anomalies(thresh+thresholds[no_fp_index,1], errors)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported) 
    results.append([0, np.mean(delays), np.std(delays), np.sum(detects)/12])

    for i in tqdm(val_range):
        # offset_index = np.abs(thresholds[:,0] - i).argmin()
        offset_index = thresholds.shape[0] - 1 - np.argmin(np.abs(thresholds[:,0][::-1] - i))
        offset = thresholds[offset_index,1]
        anomaly_pred = threshold_anomalies(thresh+offset, errors)
        delays, detects = crash_detection_delay(anomaly_pred, crash_reported) 
        if np.sum(detects) == 0:
            delays = [30]
        results.append([thresholds[offset_index,0], np.mean(delays), np.std(delays), np.sum(detects)/12])
        
    anomaly_pred = threshold_anomalies(thresh+thresholds[all_fp_index,1], errors)
    delays, detects = crash_detection_delay(anomaly_pred, crash_reported) 
    results.append([1, np.mean(delays), np.std(delays), np.sum(detects)/12])

    return results

def calculate_tp_fp(anomaly_pred, anomaly_labels):
    tps = 0
    fps = 0
    tns = 0
    num_anom = 0
    anomaly_pred = anomaly_pred.flatten()
    num_nodes = 196
    for i in range(0, len(anomaly_labels), num_nodes):
        predictions = anomaly_pred[i:i+num_nodes]
        predicted_anomaly = np.any(predictions==1)
        
        true_vals = anomaly_labels[i:i+num_nodes]
        actually_anomaly = np.all(true_vals==1)
        if np.any(true_vals==-1):
            continue 
        if predicted_anomaly and actually_anomaly:
            tps += 1
        elif predicted_anomaly and not actually_anomaly:
            fps += 1
            
        if actually_anomaly:
            num_anom += 1

        if not actually_anomaly and not predicted_anomaly:
            tns += 1
    
    tpr = tps / num_anom
    fpr = fps / (fps + tns)
    return tpr, fpr

def find_thresholds(thresh, errors, anomaly_labels):
    results = []
    for i in tqdm(np.linspace(-0.1, 0.2, 1000)):
        anomaly_pred = threshold_anomalies(thresh+i, errors)
        tpr, fpr = calculate_tp_fp(anomaly_pred, anomaly_labels)
        results.append([fpr, i])
        
    return results

def crash_detection_delay(anomaly_pred, crash_reported):
    time_anomalies = np.any(anomaly_pred==1, axis=1)
    delay = []
    detects = []
    
    reported_indices = np.where(crash_reported == 1)[0]
    for i in reported_indices:
        detected = False
        for t in range(i-30, i+30):
            if time_anomalies[t] == 1:
                delay.append(t-i)
                detected = True
                break
            
        detects.append(detected)
    
    return delay, detects

def calculate_accuracy(anomaly_pred, anomaly_labels):
    correct = []
    anomaly_pred = anomaly_pred.flatten()
    num_nodes = 196
    for i in range(0, len(anomaly_labels), num_nodes):
        predictions = anomaly_pred[i:i+num_nodes]
        predicted_anomaly = np.any(predictions==1)
        
        true_vals = anomaly_labels[i:i+num_nodes]
        actually_anomaly = np.any(true_vals==1)
        if np.any(true_vals==-1):
            continue 
        if predicted_anomaly == actually_anomaly:
            correct.append(1)
        else:
            correct.append(0)
    
    correct = np.array(correct)
    return np.sum(correct) / len(correct)

def n_in_a_row(anomalies, n):
    result = np.zeros_like(anomalies)
    for j in range(anomalies.shape[1]):
        counter = 0
        for i in range(anomalies.shape[0]):
            if anomalies[i,j] == 1:
                counter += 1
            else:
                counter = 0
            
            if counter >= n:
                result[i,j] = 1
                
    return result

def calculate_auc(test_errors, anomaly_labels):
    def anomaly_score(errors):
        return np.max(errors, axis=1)
    
    def remove_unknowable(score, anomaly_labels):
        time_anomalies = anomaly_labels.reshape(-1,196)[:,0] # they are all the same
        known = time_anomalies != -1
        return score[known], time_anomalies[known]
    
    score = anomaly_score(test_errors)
    score, time_labels = remove_unknowable(score, anomaly_labels)
    fpr, tpr, thresholds = roc_curve(time_labels, score)
    roc_auc = auc(fpr, tpr)
    return roc_auc