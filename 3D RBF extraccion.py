import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Function to reduce the number of decimals on the values. 
def decimals(A, N):
    factor = 10.0 ** N
    return np.floor(A * factor)/factor

#We reduce the data 
def reduce(matrix, indices, axis):

    return matrix.take(indices, axis=axis)

#Function to load the simulation data
def load_data(data_path):
    
    #We list all the files in the directory
    data_list = os.listdir(data_path)
    num_files = len(data_list)
    
    #We list all the files in the directory
    sample_file = os.path.join(data_path, data_list[0])
    with open(sample_file, 'r') as f:
        num_points = len(f.readlines())

    Alpha = np.zeros((num_files, 1))
    Mach = np.zeros((num_files, 1))
    Cp = np.zeros((num_points, num_files))
    xpos = np.zeros(num_points)
    ypos = np.zeros(num_points)
    zpos = np.zeros(num_points)
    
#We loop through all the files listed
    for i, file in enumerate(data_list):
        #We obtain the value of alfa and mach from the name of the file
        alpha, mach = map(float, file.replace('.dat', '').split(','))
        Alpha[i] = alpha
        Mach[i] = mach
        
        #We open the file to read it line by line
        with open(os.path.join(data_path, file), 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                #The values are structured as follows: X, Y, _, Cp. The values are separated by spaces.
                x, y, z, cp = map(float, line.split())
                #Cp values will be saved as follows. Each column will have the Cp values corresponding to a file.
                Cp[j, i] = cp 
                #X and Y are the same for all the simulations, so we only save them in the first iteration
                if i == 0:
                    xpos[j] = x
                    ypos[j] = y
                    zpos[j] = z

    return Alpha, Mach, Cp, xpos, ypos, zpos

def main():
    #Variables
    por=90
    trainCount = int(np.floor(495*por/100))
    epsilon = 95
    rank = 50
    type = 0

    #Training data
    Attack_angle_range = [0, 3.5]
    Mach_range = [0.6, 0.85]

    data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\BBDD 3D\\FOM_Skin_Data"

    Attack_angle, Mach, Cp, xpos, ypos, zpos = load_data(data_path)

    i=0
    removeMatrix =np.array([])
    while (i<len(Attack_angle)):
        if Attack_angle_range[0] < Attack_angle[i] < Attack_angle_range[1]:
            if Mach_range[0] < Mach[i] < Mach_range[1]:
                t = 0
            else:
                removeMatrix = np.append(removeMatrix, i)
        else:
            removeMatrix = np.append(removeMatrix, i)
        
        i+=1

    removeMatrix = removeMatrix.astype(int)

    Attack_angle = np.delete(Attack_angle, removeMatrix, axis = 0)
    Mach = np.delete(Mach, removeMatrix, axis=0)

    Cp = np.delete(Cp, removeMatrix, axis = 1)
    print("train count:", len(Attack_angle))
    if(len(Attack_angle)> trainCount):
        np.random.seed(42)
        selected_indices = np.random.choice(len(Attack_angle), trainCount, replace=False)
        remaining_indices = np.setdiff1d(np.arange(len(Attack_angle)), selected_indices)
        
        #Training data 80%
        Attack_angle_train = reduce(Attack_angle, selected_indices, 0)
        Mach_train = reduce(Mach, selected_indices, 0)
        Cp_train = reduce(Cp, selected_indices, 1)
        
        #Test data 20%
        Attack_angle_test = reduce(Attack_angle, remaining_indices, 0)
        Mach_test = reduce(Mach, remaining_indices, 0)
        Cp_test = reduce(Cp, remaining_indices, 1)

    parameters_train = np.column_stack((Attack_angle_train, Mach_train)) 
    parameters_train = np.round(parameters_train, 5)

    parameters_test = np.column_stack((Attack_angle_test, Mach_test)) 
    parameters_test = np.round(parameters_test, 5)

    t=0
    train = []
    while t < len(parameters_train):
        
        train.append([parameters_train[t, 0], parameters_train[t, 1]])
        t += 1
    
    i = 0
    test = []
    while i< len(parameters_test):
        
        test.append([parameters_test[i, 0], parameters_test[i, 1]])
        i += 1


    df = pd.DataFrame(train, columns=["Alfa", "Mach"])
    df.to_excel("Datos_train_20252004.xlsx", index=False)

    df_test = pd.DataFrame(test, columns=["Alfa", "Mach"])
    df_test.to_excel("Datos_test_20252004.xlsx", index=False)

if __name__ == '__main__':
    main()
    print("Done!")
