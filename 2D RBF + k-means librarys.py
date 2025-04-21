import numpy as np
import matplotlib.pyplot as pp
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#Function to reduce the amount of data. We only need 80% of the data to train the model.
#This function allows us to take the values that correspond to the indices established
def reduce_data(data, indices, axis):
    
    return data.take(indices, axis=axis)

# Function to truncate the decimals
def truncate(arr, decimals = 5):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor

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
                x, y, _, cp = map(float, line.split())
                #Cp values will be saved as follows. Each column will have the Cp values corresponding to a file.
                Cp[j, i] = cp 
                #X and Y are the same for all the simulations, so we only save them in the first iteration
                if i == 0:
                    xpos[j] = x
                    ypos[j] = y

    return Alpha, Mach, Cp, xpos, ypos

def encontrar_k_optimo(Cp_t, k_max=10):
    inercias = []
    sil_scores = []

    Ks = range(2, k_max + 1)  # comenzamos desde 2 clusters

    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(Cp_t)
        inercias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(Cp_t, labels))

    # Plot del método del codo
    pp.figure(figsize=(12, 5))

    pp.subplot(1, 2, 1)
    pp.plot(Ks, inercias, 'bo-')
    pp.xlabel('Número de Clusters (k)')
    pp.ylabel('Inercia')
    pp.title('Método del Codo')

    # Plot del coeficiente de silhouette
    pp.subplot(1, 2, 2)
    pp.plot(Ks, sil_scores, 'go-')
    pp.xlabel('Número de Clusters (k)')
    pp.ylabel('Silhouette Score')
    pp.title('Coeficiente de Silhouette')

    pp.tight_layout()
    pp.show()

    k_silhouette = Ks[np.argmax(sil_scores)]
    print(f"🔍 Mejor k según Silhouette Score: {k_silhouette}")

    return k_silhouette

AlphaRange = [0, 2]
MachRange = [0.6, 0.75]
trainCount = int(np.floor(2000*0.8))

#Values that will be used for the testing
T1 = [0.21619, 0.61428]
T1OG = [0.03294, 0.25495]
T2 = [1.42518, 0.63396]
T2OG = [0.22383, 0.2476]
T3 = [0.87830, 0.72408]
T3OG = [0.17041, 0.24089]
T4 = [1.19373, 0.74891]
T4OG = [0.2735, 0.24321]
T5 = [1.96717, 0.70797]
T5OG = [0.37336, 0.23625]


data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\Naca0012_database_mesh_1\\FOM_Skin_Data"
#We load the data from the database
Alpha, Mach, Cp, xpos, ypos = load_data(data_path)

#We truncate the Alpha and Mach values to 5 decimals
Alpha = truncate(Alpha)
Mach = truncate(Mach)

#Creation of an array with all the test samples
TestSamples = np.stack((T1, T2, T3, T4, T5))

#A range of alpha and Mach has been established. Here we are going to loop through every value of alpha and mach to eliminate the values that are outside the range defined
i = 0
removeMatrix = np.array([])
while i < len(Alpha):
    #Testing if alpha is within the range
    if AlphaRange[0] < Alpha[i] < AlphaRange[1]:
        #Testing if Mach is within the range
        if MachRange[0] < Mach[i] < MachRange[1]:
            kk = 0
        else:
            #If Mach is not in the range we add this value to the remove Matrix
            removeMatrix = np.append(removeMatrix, i)
    else:
        #If Alpha is not in the range we add this value to the remove matrix
        removeMatrix = np.append(removeMatrix, i)
    
    y = 0
    #We remove the test samples from the original alpha and Mach matrices
    while y < TestSamples.shape[0]:
        if Alpha[i] == TestSamples[y, 0] and Mach[i] == TestSamples[y, 1]:
            removeMatrix = np.append(removeMatrix, i)
        y+=1
    i+=1

#Elimination of the values outside the range
removeMatrix = removeMatrix.astype(int)
Alpha = np.delete(Alpha, removeMatrix, axis=0)
Mach = np.delete(Mach, removeMatrix, axis=0)
Cp = np.delete(Cp, removeMatrix, axis=1)



if(len(Alpha) > trainCount):
    #Thanks to this seed, every time the code is executed, the same values will be chosen in this 80%
    np.random.seed(42)
    #Selection of the indices corresponding to this 80%
    selected_indices = np.random.choice(len(Alpha), trainCount, replace=False)
    #We reduce the matrix to the 80%
    Alpha = reduce_data(Alpha, selected_indices, 0)
    Mach = reduce_data(Mach, selected_indices, 0)
    Cp = reduce_data(Cp, selected_indices, 1)

print("Samples: ", len(Alpha))

#Creation of an array will all the Alfa and Mach values.
parameters = np.column_stack((Alpha, Mach))
X_total = np.hstack([Cp.T, parameters])
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X_total)

k_max = 40
k_optimo = encontrar_k_optimo(X_normalizado, k_max)
kmeans = KMeans(n_clusters=k_optimo, random_state=0)
labels = kmeans.fit_predict(X_normalizado)
pp.figure(figsize=(8, 6))
for i in range(k_optimo):
    idx = labels == i
    pp.scatter(parameters[idx, 0], parameters[idx, 1], label=f'Cluster {i}')
    
pp.xlabel('Alpha (°)')
pp.ylabel('Mach')
pp.title('K-Means Clustering de Simulaciones')
pp.legend()
pp.grid(True)
pp.tight_layout()
pp.show()