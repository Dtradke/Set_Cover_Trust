import math
import pickle
import datetime
import pandas as pd
import csv
import time
import os


def findError(prediction, ground):
    return abs(prediction - ground)



def best_dict_to_csv(best_dict):
    today_string = time.strftime("%Y%m%d-%H%M%S")
    filename =  "best_set_covers/" + today_string + 'LSTM.csv' #model_str
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in best_dict.items():
            writer.writerow([key, value])
