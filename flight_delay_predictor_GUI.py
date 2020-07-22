import tkinter
import pandas as pd
import datetime
currentDT = datetime.datetime.now()
import json
import pprint
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter.font as tkFont

# Things I need:

# Window with 4 user inputs:
    # 


window = tkinter.Tk()
window.columnconfigure([0,6], minsize=100)
window.rowconfigure([0, 6], minsize=100)
window.title("Flight Delay Predictor")