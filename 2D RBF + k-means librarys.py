import numpy as np
import matplotlib.pyplot as pp
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

def compute_CL(cp, x_c, y_y, show, text):
    CP_lower, CP_upper = separate_Cp(cp, x_c, y_y, show, text)
    CP_lower = CP_lower[CP_lower[:, 0] < 1]
    CP_upper = CP_upper[CP_upper[:, 0] < 1]
    x_lower, cp_lower = CP_lower[:, 0], CP_lower[:, 1]
    x_upper, cp_upper = CP_upper[:, 0], CP_upper[:, 1]
    LOWER = np.trapezoid(cp_lower, x_lower)
    UPPER = np.trapezoid(cp_upper, x_upper)
    lift = LOWER - UPPER
    return lift

def compute_AC(cp, x_c, y_y, show, text):
    CP_lower, CP_upper = separate_Cp(cp, x_c, y_y, show, text)
    CP_lower = CP_lower[CP_lower[:, 0] < 1]
    CP_upper = CP_upper[CP_upper[:, 0] < 1]
    x_lower, cp_lower = CP_lower[:, 0], CP_lower[:, 1]
    x_upper, cp_upper = CP_upper[:, 0], CP_upper[:, 1]
    lift_lower = np.trapezoid(cp_lower, x_lower)
    lift_upper = np.trapezoid(cp_upper, x_upper)
    lift = lift_lower - lift_upper
    CM_le_lower = np.trapezoid(cp_lower * x_lower, x_lower)
    CM_le_upper = np.trapezoid(cp_upper * x_upper, x_upper)
    CM_le = CM_le_lower - CM_le_upper
    x_ac = CM_le / lift
    return x_ac

def separate_Cp(cp1, x1, y1, show, text):
    CPlower, xlower, CPupper, xupper = [], [], [], []
    i = 0
    while i < len(cp1):
        if x1[i] < 1:
            if y1[i] < 0:
                CPlower.append(cp1[i])
                xlower.append(x1[i])
            else:
                CPupper.append(cp1[i])
                xupper.append(x1[i])
        i += 1
    CPlower = np.array(CPlower)
    CPupper = np.array(CPupper)
    xlower = np.array(xlower)
    xupper = np.array(xupper)
    if show:
        pp.figure()
        pp.plot(xupper, CPupper, label="Upper Surface", marker="o")
        pp.plot(xlower, CPlower, label="Lower Surface", marker="x")
        pp.gca().invert_yaxis()
        pp.title("Pressure Coefficient (Cp) Distribution " + text)
        pp.xlabel("x")
        pp.ylabel("Cp")
        pp.legend()
        pp.grid()
        pp.show(block=False)
    lower_array = np.column_stack((xlower, CPlower))
    upper_array = np.column_stack((xupper, CPupper))
    return lower_array, upper_array

def computeError(ACL, OGACL):
    print("-----CL-----")
    newCL = ACL[0]
    ogCL = OGACL[0]
    CL_abs_error = truncate(np.abs(ogCL - newCL))
    CL_rel_error = truncate(CL_abs_error / ogCL * 100)
    print(str(CL_abs_error))
    print(str(CL_rel_error), "%")
    print("-----AC-----")
    newAC = ACL[1]
    ogAC = OGACL[1]
    AC_abs_error = truncate(np.abs(ogAC - newAC))
    AC_rel_error = truncate(AC_abs_error / ogAC * 100)
    print(str(AC_abs_error))
    print(str(AC_rel_error), "%")
    return np.array([[CL_abs_error, CL_rel_error], [AC_abs_error, AC_rel_error]])

def read_filename(filename):
    try:
        Cp_values = np.loadtxt(filename, delimiter=";")
        return Cp_values
    except Exception as e:
        print(f"Error al leer el fichero {filename} : {e}")
        return None
    
def compute_Cp_error(Cp_real, Cp_interp):
    abs_error = np.abs(Cp_real - Cp_interp)
    rel_error = (abs_error / np.abs(Cp_real)) * 100
    sim_error = np.linalg.norm(Cp_real - Cp_interp) / np.linalg.norm(Cp_real)
    sim_error = truncate(sim_error)
    print("Error global de la simulación: " + str(sim_error))
    error_squared = (Cp_real - Cp_interp) ** 2
    rmse = np.sqrt(np.mean(error_squared))
    rmse = truncate(rmse)
    print("Error cuadrático medio: " + str(rmse))
    error_abs_prom = np.mean(np.abs(Cp_real - Cp_interp))
    error_abs_prom = truncate(error_abs_prom)
    print(f"Error absoluto promedio: {error_abs_prom}")
    return abs_error, rel_error

def plot_cp(Cp_real, Cp_interpolado, text):
    pp.figure()
    pp.scatter(Cp_real, Cp_interpolado, color='blue', label='Cp interpolado vs Cp real')
    pp.plot([min(Cp_real), max(Cp_real)], [min(Cp_real), max(Cp_real)], color='red', linestyle='--', label='Interpolación perfecta (y = x)')
    pp.xlabel('Cp real')
    pp.ylabel('Cp interpolado')
    pp.title('Comparación de Cp real vs Cp interpolado ' + text)
    pp.legend()
    pp.grid(True)

def train_cluster_classifier(features, cluster_labels, n_neighbors=5):

    # Usamos solo (alfa, mach) para clasificación
    X = features[:, -2:]  # columnas alfa y mach
    y = cluster_labels
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)
    return clf

def assign_cluster(classifier, test_point):

    test_point_normalized = scaler_inputs.transform(test_point)
    cluster_label = classifier.predict(test_point_normalized)
    return cluster_label[0]

def train_rom_clusters(parameters, Cp, n_clusters, epsilon, rank, kernel, cluster_labels):

    rom_clusters = {}
    # Para cada cluster, extraemos los índices y entrenamos un ROM
    for cluster in range(n_clusters):
        indices = np.where(cluster_labels == cluster)[0]
        if len(indices) == 0:
            continue
        params_cluster = parameters[indices, :]
        Cp_cluster = Cp[:, indices]  # columnas correspondientes

        # Se crea la base de datos y se entrena el modelo ROM para el cluster
        db_cluster = Database(params_cluster, Cp_cluster.T)
        pod_cluster = POD('svd', rank=Cp_cluster.shape[1] if Cp_cluster.shape[1] < rank else rank)
        rbf_cluster = RBF(kernel=kernel, epsilon=epsilon)
        rom_cluster = ROM(db_cluster, pod_cluster, rbf_cluster)
        rom_cluster.fit()
        rom_clusters[cluster] = rom_cluster
        print(f"Cluster {cluster}: {len(indices)} muestras.")
    return rom_clusters

def predict_kmeans(test_sample, OGACL, xpos, ypos, kmeans, rom_clusters, clas):


    label = assign_cluster(clas, test_sample)
    # Determinar la etiqueta del cluster para la muestra de prueba
    
    print(f"Test sample {test_sample} asignado al cluster: {label}")
    
    # Predecir usando el ROM del cluster asignado
    rom = rom_clusters[label]
    newCP = rom.predict(np.array(test_sample)).snapshots_matrix  # Cp interpolados
    newCP = newCP.T
    
    # Cargar datos reales para comparar
    name = 'C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\' + f"Cp_Alfa_{test_sample[0, 0]}_Mach_{test_sample[0, 1]}.txt"
    Cp_real = read_filename(name)
    abs_error, rel_error = compute_Cp_error(Cp_real, newCP)
    text = "A = " + str(test_sample[0, 0]) + " M = " + str(test_sample[0, 1])
    plot_cp(Cp_real, newCP, text)
    t = "New: A = " + str(test_sample[0, 0]) + " M = " + str(test_sample[0, 1])
    Cl = compute_CL(newCP, xpos, ypos, True, t)
    AC = compute_AC(newCP, xpos, ypos, False, t)
    Cl = truncate(Cl)
    AC = truncate(AC)
    print(" CL  = ", Cl, " AC  = ", AC)
    print("OGCL = ", OGACL[0], "OGAC = ", OGACL[1])
    computeError([Cl, AC], OGACL)

# def encontrar_k_optimo(Cp_t, k_max=10):
#     inercias = []
#     sil_scores = []

#     Ks = range(2, k_max + 1)  # comenzamos desde 2 clusters

#     for k in Ks:
#         kmeans = KMeans(n_clusters=k, random_state=0)
#         labels = kmeans.fit_predict(Cp_t)
#         inercias.append(kmeans.inertia_)
#         sil_scores.append(silhouette_score(Cp_t, labels))

#     # Plot del método del codo
#     pp.figure(figsize=(12, 5))

#     pp.subplot(1, 2, 1)
#     pp.plot(Ks, inercias, 'bo-')
#     pp.xlabel('Número de Clusters (k)')
#     pp.ylabel('Inercia')
#     pp.title('Método del Codo')

#     # Plot del coeficiente de silhouette
#     pp.subplot(1, 2, 2)
#     pp.plot(Ks, sil_scores, 'go-')
#     pp.xlabel('Número de Clusters (k)')
#     pp.ylabel('Silhouette Score')
#     pp.title('Coeficiente de Silhouette')

#     pp.tight_layout()
#     pp.show()

#     k_silhouette = Ks[np.argmax(sil_scores)]
#     print(f"🔍 Mejor k según Silhouette Score: {k_silhouette}")

#     return k_silhouette

trainCount = int(np.floor(2000 * 0.8))
epsilon = 95
rank = 50
kernel = 'linear'
AlphaRange = [0, 2]
MachRange = [0.6, 0.75]
n_clusters = 5

#Values that will be used for the testing
T1 = np.array([[0.21619, 0.61428]])
T1OG = [0.03294, 0.25495]
T2 = np.array([[1.42518, 0.63396]])
T2OG = [0.22383, 0.2476]
T3 = np.array([[0.87830, 0.72408]])
T3OG = [0.17041, 0.24089]
T4 = np.array([[1.19373, 0.74891]])
T4OG = [0.2735, 0.24321]
T5 = np.array([[1.96717, 0.70797]])
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


kmeans = KMeans(n_clusters, random_state=0)
labels = kmeans.fit_predict(X_normalizado)
pp.figure(figsize=(8, 6))
for i in range(n_clusters):
    idx = labels == i
    pp.scatter(parameters[idx, 0], parameters[idx, 1], label=f'Cluster {i}')
    
pp.xlabel('Alpha (°)')
pp.ylabel('Mach')
pp.title('K-Means Clustering de Simulaciones')
pp.legend()
pp.grid(True)
pp.tight_layout()

scaler_inputs = StandardScaler()
X_inputs = np.column_stack((Alpha, Mach))  # shape (n_samples, 2)
X_inputs_normalized = scaler_inputs.fit_transform(X_inputs)

clas = train_cluster_classifier(X_inputs_normalized, labels, n_neighbors=1)
rom_clusters = train_rom_clusters(parameters, Cp, n_clusters, epsilon, rank, kernel, labels)

# Predicciones para las muestras de prueba
print("--------------T1--------------")
predict_kmeans(T1, T1OG, xpos, ypos, kmeans, rom_clusters, clas)
print("--------------T2--------------")
predict_kmeans(T2, T2OG, xpos, ypos, kmeans, rom_clusters, clas)
print("--------------T3--------------")
predict_kmeans(T3, T3OG, xpos, ypos, kmeans, rom_clusters, clas)
print("--------------T4--------------")
predict_kmeans(T4, T4OG, xpos, ypos, kmeans, rom_clusters, clas)
print("--------------T5--------------")
predict_kmeans(T5, T5OG, xpos, ypos, kmeans, rom_clusters, clas)

pp.show()