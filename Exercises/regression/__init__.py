# This is a sample Python script.
import sys
import tensorflow as tf
import sklearn
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib as plot

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print(sys.version)
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    print('Tensorflow version is {}.'.format(tf.__version__))
    print('The Pandas version is {}.'.format(pd.__version__))
    print('The Numpy version is {}.'.format(np.__version__))
    print('The Seaborn version is {}.'.format(sb.__version__))
    print('The Matplotlib version is {}.'.format(plot.__version__))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

