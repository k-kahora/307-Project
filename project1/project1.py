import numpy as np
import pandas as pd
from ISLP import load_data
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

AUTO = load_data("Auto")
selected_coloumns = ['mpg','cylinders', 'displacement',  'weight', 'acceleration', ]
origin = ['origin']

print(AUTO[origin])
print(AUTO[selected_coloumns].head)
print(AUTO[selected_coloumns])
