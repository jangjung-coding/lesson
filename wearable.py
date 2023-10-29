import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'raw_data_Liu.csv'

df = pd.read_csv(path)
df.head()