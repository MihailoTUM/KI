import numpy as np
import pandas as pd
import kagglehub

class DataLoader():
    def __init__(self, batch=32):
        self.batch = batch

    
path = kagglehub.dataset_download("uciml/iris")

data = pd.read_csv(f"{path}/Iris.csv")
print(data.head())