import os

def append(directory, label):
    # Append new data points to csv
    print('write files')
    for dirpaths, dirnames, filenames in os.walk(directory):
        with open('./dataset.csv', 'a') as f:
            for filename in filenames:
                f.write(f'{filename},{label}\n')

directory = '/home/cosc/student/bmc142/dataset/predator_dataset/audio/'
label = 1
append(directory, label)
