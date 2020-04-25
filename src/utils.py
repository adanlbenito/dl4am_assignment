import pandas as pd
import os
import pickle

def load_dataframe(name, path=None):
    loc = './' if not path else path 
    file = loc+name
    dataframe = None
    is_loaded = False
    if os.path.isfile(file):
        with open(file, "rb") as f:
            try:
                dataframe = pd.read_pickle(file)
                is_loaded = True
                print('Data loaded successfully')
            except Exception: 
                print('Could not load data')
                pass
    return is_loaded, dataframe

