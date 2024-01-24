#extracts information from tensorboard logs and writes into pd dataframe

import os
import pandas as pd
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
import logging

logging.basicConfig(filename="/home/kandelaki/git/SAM-Adapter-PyTorch/figures/extractor.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.ERROR)

import tensorboard.backend.event_processing.event_accumulator

metrics = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy', 'F1', 'AUCROC']

# Adjusts folder names by ceiling floating point numbers
def rename_folders(path):
    #rename folders to remove spaces
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.startswith("tested_on"):
               factor = dir.split("_")[-1]
               factor = float("{:.2f}".format(float(factor)))
               os.rename(os.path.join(root, dir), os.path.join(root, "tested_on_"+str(factor)))             
               
# def read_mean_values_from_log(path):

def take_mean_value_of_metric(path, metric):
    x = EventMultiplexer(size_guidance={
        tensorboard.backend.event_processing.event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        tensorboard.backend.event_processing.event_accumulator.IMAGES: 1,
        tensorboard.backend.event_processing.event_accumulator.AUDIO: 1,
        tensorboard.backend.event_processing.event_accumulator.SCALARS: 0,
        tensorboard.backend.event_processing.event_accumulator.HISTOGRAMS: 1,
    }).AddRunsFromDirectory(path = path)
    x.Reload()
    try:
        scalars = x.Scalars(tag=metric, run='mean')
    except:
        logging.error("No scalars found for metric: "+metric+" in path: "+path)
        return None
    mean = scalars[-1]
    max_value = max(scalars, key=lambda x: x.value)
    min_value = min(scalars, key=lambda x: x.value)
    max_step = max_value.step
    min_step = min_value.step
    return {"mean": mean.value, "max_value": max_value.value, "min_value": min_value.value, "max_step": max_step, "min_step": min_step}

def create_dataframes_from_tensorboard_logs(path):
    os.makedirs('dataframes', exist_ok=True)

    rename_folders(path)

    for dirpath, dirnames, filenames in os.walk(path):
        for dir in dirnames:
            if dir == "test":
                trained_on = dirpath.split("/")[-3]
                trained_on_factor = trained_on.split("_")[-1]
                trained_on_factor = float("{:.2f}".format(float(trained_on_factor)))
                tested_on = dirpath.split("/")[-1]
                tested_on_factor = tested_on.split("_")[-1]
                tested_on_factor = float("{:.2f}".format(float(tested_on_factor)))
                dataset = dirpath.split("/")[-2]

                for metric in metrics:
                    path_to_metric = os.path.join(dirpath, dir, metric)
                    values_dict = take_mean_value_of_metric(path_to_metric, metric)
                    if values_dict is None:
                        continue
                    mean = values_dict["mean"]
                    max_value = values_dict["max_value"]
                    min_value = values_dict["min_value"]
                    max_step = values_dict["max_step"]
                    min_step = values_dict["min_step"]  

                    df = pd.DataFrame({"trained_on": [trained_on_factor], "tested_on": [tested_on_factor], "dataset": [dataset], "metric": [metric], "mean": [mean], "max_value": [max_value], "min_value": [min_value], "max_step": [max_step], "min_step": [min_step]})
                    # Save everything to one csv 
                    if os.path.isfile('dataframes/df.csv'):
                        df.to_csv('dataframes/df.csv', mode='a', header=False, index=False)
                    else:
                        df.to_csv('dataframes/df.csv', mode='a', header=True, index=False)


create_dataframes_from_tensorboard_logs('/home/kandelaki/git/SAM-Adapter-PyTorch/cross_test/')