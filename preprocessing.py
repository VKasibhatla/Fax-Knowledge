import os
import numpy as np

def parse():
    with open('data.txt') as f:
        lines = f.readlines()
    data = [line.split(',')[1:3] for line in lines]
    return data
