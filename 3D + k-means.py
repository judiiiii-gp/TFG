import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.spatial import Delaunay, ConvexHull
from scipy.interpolate import griddata
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
import matplotlib.tri as mtri
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score


#Function to reduce the amount of data. We only need 80% of the data to train the model.
#This function allows us to take the values that correspond to the indices established
def reduce_data(data, indices, axis):
    return data.take(indices, axis=axis)


# Function to truncate the decimals
def truncate(arr, decimals=5):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor

# Function to calculate the error in the lift coefficient and in the aerodynamic centre
def computeError(ACL, OGACL):

    print("-----CL-----")

    newCL = ACL[0]
    ogCL = OGACL[0]

    #Calculation of the absolute error and truncation of the result
    CL_abs_error = truncate(np.abs(ogCL - newCL))
    #Calculation of the relative error and truncation of the result
    CL_rel_error = truncate((CL_abs_error / ogCL) * 100)

    print(str(CL_abs_error))
    print(str(CL_rel_error), "%")

    print("-----AC-----")

    newAC = ACL[1]
    ogAC = OGACL[1]

    #Same calculation as the one made for the lift coefficient
    AC_abs_error = truncate(np.abs(ogAC - newAC))
    AC_rel_error = truncate((AC_abs_error / ogAC) * 100)

    print(str(AC_abs_error))
    print(str(AC_rel_error), "%")

    #We return an array with the error values
    return np.array([[CL_abs_error, CL_rel_error], [AC_abs_error, AC_rel_error]])

def separate_Cp(xpos, ypos, zpos, Cp):
    i = 0
    x_fom_up = []
    y_fom_up = []
    z_fom_up = []
    cp_fom_up = []
    x_fom_down = []
    y_fom_down = []
    z_fom_down = []
    cp_fom_down = []
    while i< len(zpos):
        if (zpos[i]>0):
            x_fom_up.append(xpos[i])
            y_fom_up.append(ypos[i])
            z_fom_up.append(zpos[i])
            cp_fom_up.append(Cp[i])
        else:
            x_fom_down.append(xpos[i])
            y_fom_down.append(ypos[i])
            z_fom_down.append(zpos[i])
            cp_fom_down.append(Cp[i])
        i += 1
    x_fom_up = np.array(x_fom_up).squeeze() 
    y_fom_up = np.array(y_fom_up).squeeze() 
    z_fom_up = np.array(z_fom_up).squeeze() 
    cp_fom_up = np.array(cp_fom_up).squeeze() 
    x_fom_down = np.array(x_fom_down).squeeze() 
    y_fom_down = np.array(y_fom_down).squeeze() 
    z_fom_down = np.array(z_fom_down).squeeze() 
    cp_fom_down = np.array(cp_fom_down).squeeze() 
    
    return x_fom_up, y_fom_up, z_fom_up, cp_fom_up, x_fom_down, y_fom_down, z_fom_down, cp_fom_down
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


def plot_cp_3d(xpos, ypos, zpos, Cp, Alpha, Mach):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xpos, ypos, zpos, c=Cp, cmap='viridis')
    fig.colorbar(sc, ax=ax, label='Cp')
    ax.set_title(f'Cp distribution (Alpha={Alpha}, Mach={Mach})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=135) 

    ax.set_xlim(xpos.min(), xpos.max())
    ax.set_ylim(ypos.min(), ypos.max())
    ax.set_zlim([-0.25, 0.25])
    plt.tight_layout()
    
def plot_cp_2d(xpos, ypos, Cp, Alpha, Mach):

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 3
    
    norm = plt.Normalize(vmin=Cp.min(), vmax=Cp.max())
    mask = ypos > 0 
    xpos = xpos[mask]
    ypos = ypos[mask]
    Cp = Cp[mask]
    # 1) Triangulamos TODOS los puntos (sin filtrar antes)
    triang = mtri.Triangulation(xpos, ypos)

    # 2) Enmascaramos triángulos por debajo de y=0 (opcional)
    # bar_y = np.mean(ypos[triang.triangles], axis=1)
    # triang.set_mask(bar_y <= 0)

    # 3) Pintamos con shading='gouraud' (interpolación continua)
    fig, ax = plt.subplots(figsize=(8,8))
    tpc = ax.tripcolor(triang,Cp.squeeze(),norm = norm ,cmap='viridis', edgecolors='none', linewidth = 0.8)

    # opcional: contornos suaves
    levels = np.linspace(Cp.min(), Cp.max(), 60)
    ax.tricontourf(triang, Cp.squeeze(), levels=levels, cmap='viridis', alpha=0.0)

    # colorbar y estética
    cbar = fig.colorbar(tpc, ax=ax, label='PRESSURE COEFFICIENT')
    math_label_fontsize = 14
    cbar.ax.tick_params(labelsize=math_label_fontsize)
    x_label = r'$x$'
    y_label = r'$y$'
    plt.xlabel(x_label, fontsize = math_label_fontsize)
    plt.ylabel(y_label, fontsize = math_label_fontsize)
    plt.title(f'Cp distribution (Alpha={Alpha}, Mach={Mach})')
    plt.tight_layout()

def plot_cp_section(xpos, ypos, zpos, Cp, Cp_real, parameters):
    labels = [90, 65, 20]

    x_fom_up, y_fom_up, z_fom_up, cp_fom_up, x_fom_down, y_fom_down, z_fom_down, cp_fom_down = separate_Cp(xpos, ypos, zpos, Cp)
    x_fom_up_real, y_fom_up_real, z_fom_up_real, cp_fom_up_real, x_fom_down_real, y_fom_down_real, z_fom_down_real, cp_fom_down_real = separate_Cp(xpos, ypos, zpos, Cp_real)
    for section in range(3):
        fig = plt.figure(figsize=(4, 3))
        
        # SET Y POSITION TO PLOT CP DISTRIBUTION
        b = 1.1963
        y_target = labels[section]*b/100 # y_target = y/b
        # CREATE X LINEAR SPACE
        x_min = (y_target)*np.tan(30*np.pi/180)             + 0.001
        x_max = (y_target)*np.tan(15.8*np.pi/180)+0.8059    - 0.01
        x_grid = np.linspace(x_min, x_max, 250)
        # SET X BETWEEN 0 and 1
        x_airfoil_normalized = (x_grid - x_min) / (x_max - x_min)
        x_airfoil_normalized_full = np.concatenate((x_airfoil_normalized, x_airfoil_normalized[::-1]))
        
        cp_interpolated_fom = griddata((x_fom_up, y_fom_up), cp_fom_up, (x_grid,y_target), method='linear', fill_value=0.25)
        cp_interpolated_fom_inf = griddata((x_fom_down, y_fom_down), cp_fom_down, (x_grid,y_target), method='linear', fill_value=0.25)
        cp_interpolated_fom_full = np.concatenate((cp_interpolated_fom, cp_interpolated_fom_inf[::-1]))
        plt.plot(x_airfoil_normalized_full, -cp_interpolated_fom_full, '-', color='blue', marker = "o", linewidth=1.5, label='Interpolated Cp')

        # cp_interpolated_fom_real = griddata((x_fom_up_real, y_fom_up_real), cp_fom_up_real, (x_grid,y_target), method='linear', fill_value=0.25)
        # cp_interpolated_fom_inf_real = griddata((x_fom_down_real, y_fom_down_real), cp_fom_down_real, (x_grid,y_target), method='linear', fill_value=0.25)
        # cp_interpolated_fom_full_real = np.concatenate((cp_interpolated_fom_real, cp_interpolated_fom_inf_real[::-1]))
        # plt.plot(x_airfoil_normalized_full, -cp_interpolated_fom_full_real, '-', color='green', linewidth=1.5, label='Real Cp')
        
        # plt.xlim(-0.1, 1.1)
        # plt.ylim(-1.0, 1.3)

        # Set labels
        math_label_fontsize = 14
        x_label = r'$x$'
        y_label = r'$-C_p$'
        plt.xlabel(x_label, fontsize = math_label_fontsize)
        plt.ylabel(y_label, fontsize = math_label_fontsize)
        plt.title("Plot section y = " + str(labels[section]) + " \nAlfa = " + str(parameters[0,0]) + " Mach = " + str(parameters[0, 1]))
        plt.legend(loc='upper right')
        plt.tight_layout()

#Function that performs the interpolation
def predict(ACL, rom):

    newCP = rom.predict(ACL).snapshots_matrix # interpolated values
    newCP = newCP.T


    return newCP

def calcular_cl_desde_puntos(coords, Cp, alfa_deg, mach, rho=1.225, T=288.15,gamma=1.4, R=287.05):


    # 1) PCA para definir el plano principal del ala
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eigh(cov)
    # orden descendente de varianza
    idx = np.argsort(vals)[::-1]
    e1, e2 = vecs[:, idx[0]], vecs[:, idx[1]]

    # 2) Proyección 2D
    pts2d = np.stack([centered.dot(e1), centered.dot(e2)], axis=1)

    # 3a) Triangulación para integración
    tri = Delaunay(pts2d)
    tris = tri.simplices

    # 3b) Área de referencia = área del hull en 2D
    hull = ConvexHull(pts2d)
    S = hull.volume  # en 2D, volume = área

    # 4) Condiciones de flujo
    a = np.sqrt(gamma * R * T)
    V = mach * a
    q_inf = 0.5 * rho * V**2
    alfa = np.radians(alfa_deg)
    lift_dir = np.array([-np.sin(alfa), 0.0, np.cos(alfa)])  # eje z' del ala

    # 5) Integración sobre cada triángulo
    L_total = 0.0
    for tri_pts in tris:
        i0, i1, i2 = tri_pts
        p0, p1, p2 = coords[i0], coords[i1], coords[i2]
        cn = np.cross(p1 - p0, p2 - p0)           # 2·área·normal
        area = 0.5 * np.linalg.norm(cn)           # área del triángulo
        n_hat = cn / (2 * area)       # normal unitaria
        
        if np.dot(n_hat, lift_dir) < 0:
            n_hat = -n_hat
            
        cp_avg = (Cp[i0] + Cp[i1] + Cp[i2]) / 3.0
        dF = - cp_avg * q_inf * area * n_hat      # fuerza elemental
        L_total += np.dot(dF, lift_dir)

    # 6) Coeficiente de sustentación
    Cl = L_total / (q_inf * S)
    print(f"Ángulo de ataque: {alfa_deg}°  |  Mach: {mach}")
    print(f"Área ala proyectada (S): {S:.3f} m²")
    print(f"q_inf: {q_inf:.2f} Pa  |  V: {V:.2f} m/s")
    print(f"Lift total: {L_total:.2f} N")
    print(f"CL calculado: {Cl:.4f}")
    print(f"Cp: min={Cp.min():.3f}, max={Cp.max():.3f}, mean={Cp.mean():.3f}")

    return Cl

def computeError(Cl_real, Cl_interp):


    #Calculation of the absolute error and truncation of the result
    CL_abs_error = truncate(np.abs(Cl_real - Cl_interp))
    #Calculation of the relative error and truncation of the result
    CL_rel_error = truncate((CL_abs_error / Cl_real) * 100)

    print("Error absoluto: " + str(CL_abs_error))
    print("Error relativo: " + str(CL_rel_error), "%")


def find_index(arr, alfa, mach, Cp):
    idx = np.where((arr[:, 0] == alfa) & (arr[:, 1] == mach))[0]
    cp_case = Cp[:, idx] 
    return cp_case

# Function to calculate the error between the Cp from the simulation and the Cp interpolated
def compute_Cp_error(Cp_real, Cp_interp):
    
    #Calculation of the absolute error of each point. We get a vector that has the absolute error of each value of Cp
    abs_error = np.abs(Cp_real - Cp_interp)
    #Calculation of the relative error of each point. We get a vector that has the relative error of each value of Cp
    rel_error = (abs_error / np.abs(Cp_real)) * 100

    #Calculation of the simulation error by obtaining the module of the relative error's vector, and dividing it by the module of the real Cp.
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

def integrate_surface(x, y, z, cp, is_upper):

    pts2d = np.vstack((x, y)).T
    tri = Delaunay(pts2d)

    lift = 0.0
    for tri_idx in tri.simplices:
        # puntos en 3D
        p1 = np.array([x[tri_idx[0]], y[tri_idx[0]], z[tri_idx[0]]])
        p2 = np.array([x[tri_idx[1]], y[tri_idx[1]], z[tri_idx[1]]])
        p3 = np.array([x[tri_idx[2]], y[tri_idx[2]], z[tri_idx[2]]])

        # vector-área (módulo del área en z)
        dA_vec = np.cross(p2 - p1, p3 - p1) / 2.0
        area = abs(dA_vec[2])

        cp_avg = cp[tri_idx].mean()

        # signo según superficie:
        # - en la cara superior, dA_z > 0 orienta normal hacia +z,
        #   y la fuerza es -Cp * area
        # - en la cara inferior, normal apunta hacia -z,
        #   y la fuerza de sustentación (hacia +z) es +Cp * area
        if is_upper:
            lift += -cp_avg * area
        else:
            lift +=  cp_avg * area

    return lift

def compute_cl(x, y, z, cp, S_ref):

    # Separa tuplas (x_up, y_up, z_up, cp_up), (x_low, y_low, z_low, cp_low)
    x_up,  y_up,  z_up,  cp_up, \
    x_low, y_low, z_low, cp_low = separate_Cp(x, y, z, cp)

    # Integral en cada superficie
    L_up   = integrate_surface(x_up,  y_up,  z_up,  cp_up, True)
    L_low  = integrate_surface(x_low, y_low, z_low, cp_low, False)

    # Lift neto y CL
    L_net  = L_up + L_low
    CL     = L_net / S_ref
    CL = truncate(CL)
    print("Cl = " + str(CL))
    return CL

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
    
    return newCP
    
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
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(Ks, inercias, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')

    # Plot del coeficiente de silhouette
    plt.subplot(1, 2, 2)
    plt.plot(Ks, sil_scores, 'go-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Coeficiente de Silhouette')

    plt.tight_layout()
    plt.show()

    k_silhouette = Ks[np.argmax(sil_scores)]
    print(f"🔍 Mejor k según Silhouette Score: {k_silhouette}")

    return k_silhouette

def plotCp(Cp_real, Cp_inter, text):
    plt.figure()
    plt.scatter(Cp_real, Cp_inter, color='blue', label='Cp interpolado vs Cp real')
    plt.plot([min(Cp_real), max(Cp_real)], [min(Cp_real), max(Cp_real)], color='red', linestyle='--', label='Interpolación perfecta (y = x)')
    plt.xlabel('Cp real')
    plt.ylabel('Cp interpolado')
    plt.title('Comparación de Cp real vs Cp interpolado ' + text)
    plt.legend()
    plt.grid(True)


    
data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\BBDD 3D\\FOM_Skin_Data"
rank = 50
epsilon = 10
kernel = 'gaussian'
AlphaRange = [0, 3.5]
MachRange = [0.6, 0.85]
S=0.7532
n_clusters = 3
T1 = np.array([[0.15232, 0.60535]])
Cl_1 = 0.01168
T2 = np.array([[1.0615, 0.62832]])
Cl_2 = 0.07651
T3 = np.array([[2.15525, 0.73018]])
Cl_3 = 0.16625
T4 = np.array([[2.89353, 0.79088]])
Cl_4 = 0.23897
T5 = np.array([[3.44724, 0.80288]])
Cl_5 = 0.29121

Alpha, Mach, Cp, xpos, ypos, zpos = load_data(data_path)
    

#Creation of an array with all the test samples
TestSamples = np.stack((T1.flatten(), T2.flatten(), T3.flatten(), T4.flatten(), T5.flatten()))
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
    
#Creation of an array will all the Alfa and Mach values.
parameters = np.column_stack((Alpha, Mach))

parameters = np.round(parameters, 5)
Cp_1 = find_index(parameters, float(T1[0,0]), float(T1[0,1]), Cp)
Cp_2 = find_index(parameters, float(T2[0,0]), float(T2[0,1]), Cp)
Cp_3= find_index(parameters, float(T3[0,0]), float(T3[0,1]), Cp)
Cp_4 = find_index(parameters, float(T4[0,0]), float(T4[0,1]), Cp)
Cp_5 = find_index(parameters, float(T5[0,0]), float(T5[0,1]), Cp)
#Number of parameters that will be used for the training. We are going to use an 80% of all the parameters.
trainCount = int(np.floor(parameters.shape[0] * 0.9))

if parameters.shape[0] > trainCount:
    #Thanks to this seed, every time the code is executed, the same values will be chosen in this 80%
    np.random.seed(42)
    #Selection of the indices corresponding to this 80%
    selected_indices = np.random.choice(parameters.shape[0], trainCount, replace=False)
    #We reduce the matrix to the 80%
    parameters = reduce_data(parameters, selected_indices, 0)
    Cp = reduce_data(Cp, selected_indices, 1)


print("Samples:", parameters.shape[0])

X_total = np.hstack([Cp.T, parameters])
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X_total)

kmeans = KMeans(n_clusters, random_state=0)
labels = kmeans.fit_predict(X_normalizado)

plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    idx = labels == i
    plt.scatter(parameters[idx, 0], parameters[idx, 1], label=f'Cluster {i}')
        
plt.xlabel('Alpha (°)')
plt.ylabel('Mach')
plt.title('K-Means Clustering de Simulaciones')
plt.legend()
plt.grid(True)
plt.tight_layout()

scaler_inputs = StandardScaler()
X_inputs = parameters  # shape (n_samples, 2)
X_inputs_normalized = scaler_inputs.fit_transform(X_inputs)

clas = train_cluster_classifier(X_inputs_normalized, labels, n_neighbors=1)
rom_clusters = train_rom_clusters(parameters, Cp, n_clusters, epsilon, rank, kernel, labels)
    
#Testing phase, where we perform the interpolation and the calculation fo the errors for each sample.
print("------ TEST 1 ------")
start_time1 = time.perf_counter()
reconstructed_cp = predict_kmeans(T1, Cl_1, xpos, ypos, kmeans, rom_clusters, clas)
end_time1 = time.perf_counter()
elapsed_time1 = end_time1 - start_time1
print("Tiempo de ejecucion: " + str(elapsed_time1))
#We return the interpolated values to the original column space
coords = np.column_stack((xpos, ypos, zpos))
#plot_cp_section(xpos, ypos, zpos, reconstructed_cp, Cp_1, T1)
plot_cp_3d(xpos, ypos, zpos, reconstructed_cp, T1[0,0], T1[0,1])
# plot_cp_2d(xpos, ypos, reconstructed_cp, T1[0,0], T1[0,1])
#plotCp(Cp_1, reconstructed_cp, "Alfa = " + str(T1[0,0]) + " Mach = " + str(T1[0,1]))
compute_Cp_error(Cp_1, reconstructed_cp)
Cl = compute_cl(xpos, ypos, zpos, reconstructed_cp, S)
computeError(Cl_1, Cl)
    
# plot_cp_3d(xpos, ypos, zpos, reconstructed_cp, T1[0, 0], T1[0, 1]) 

print("------ TEST 2 ------")
start_time2 = time.perf_counter()
reconstructed_cp = predict_kmeans(T2, Cl_2, xpos, ypos, kmeans, rom_clusters, clas)
end_time2 = time.perf_counter()
elapsed_time2 = end_time2 - start_time2
print("Tiempo de ejecucion: " + str(elapsed_time2))
#We return the interpolated values to the original column space
coords = np.column_stack((xpos, ypos, zpos))

#plot_cp_section(xpos, ypos, zpos, reconstructed_cp, Cp_2, T2)
plot_cp_3d(xpos, ypos, zpos, reconstructed_cp, T2[0,0], T2[0,1])
# plot_cp_2d(xpos, ypos, reconstructed_cp, T2[0,0], T2[0,1])
#plotCp(Cp_2, reconstructed_cp, "Alfa = " + str(T2[0,0]) + " Mach = " + str(T2[0,1]))
compute_Cp_error(Cp_2, reconstructed_cp)
Cl = compute_cl(xpos, ypos, zpos, reconstructed_cp, S)
computeError(Cl_2, Cl)
    
    
print("------ TEST 3 ------")
start_time3 = time.perf_counter()
reconstructed_cp = predict_kmeans(T3, Cl_3, xpos, ypos, kmeans, rom_clusters, clas)
end_time3 = time.perf_counter()
elapsed_time3 = end_time3 - start_time3
print("Tiempo de ejecucion: " + str(elapsed_time3))
#We return the interpolated values to the original column space
coords = np.column_stack((xpos, ypos, zpos))
#plot_cp_section(xpos, ypos, zpos, reconstructed_cp, Cp_3, T3)
plot_cp_3d(xpos, ypos, zpos, reconstructed_cp, T3[0,0], T3[0,1])
# plot_cp_2d(xpos, ypos, reconstructed_cp, T3[0,0], T3[0,1])
#plotCp(Cp_3, reconstructed_cp, "Alfa = " + str(T3[0,0]) + " Mach = " + str(T3[0,1]))
compute_Cp_error(Cp_3, reconstructed_cp)
Cl = compute_cl(xpos, ypos, zpos, reconstructed_cp, S)
computeError(Cl_3, Cl)
    
    
    
print("------ TEST 4 ------")
start_time4 = time.perf_counter()
reconstructed_cp = predict_kmeans(T4, Cl_4, xpos, ypos, kmeans, rom_clusters, clas)
end_time4 = time.perf_counter()
elapsed_time4 = end_time4 - start_time4
print("Tiempo de ejecucion: " + str(elapsed_time4))
#We return the interpolated values to the original column space
coords = np.column_stack((xpos, ypos, zpos))

#plot_cp_section(xpos, ypos, zpos, reconstructed_cp, Cp_4, T4)
plot_cp_3d(xpos, ypos, zpos, reconstructed_cp, T4[0,0], T4[0,1])
# plot_cp_2d(xpos, ypos, reconstructed_cp, T4[0,0], T4[0,1])
#plotCp(Cp_4, reconstructed_cp, "Alfa = " + str(T4[0,0]) + " Mach = " + str(T4[0,1]))
compute_Cp_error(Cp_4, reconstructed_cp)
Cl = compute_cl(xpos, ypos, zpos, reconstructed_cp, S)
computeError(Cl_4, Cl)    
    
    
print("------ TEST 5 ------")
start_time5 = time.perf_counter()
reconstructed_cp = predict_kmeans(T5, Cl_5, xpos, ypos, kmeans, rom_clusters, clas)
end_time5 = time.perf_counter()
elapsed_time5 = end_time5 - start_time5
print("Tiempo de ejecucion: " + str(elapsed_time5))
#We return the interpolated values to the original column space
coords = np.column_stack((xpos, ypos, zpos))
    
#plot_cp_section(xpos, ypos, zpos, reconstructed_cp, Cp_5, T5)
plot_cp_3d(xpos, ypos, zpos, reconstructed_cp, T5[0,0], T5[0,1])
# plot_cp_2d(xpos, ypos, reconstructed_cp, T5[0,0], T5[0,1])
#plotCp(Cp_5, reconstructed_cp, "Alfa = " + str(T5[0,0]) + " Mach = " + str(T5[0,1]))
compute_Cp_error(Cp_5, reconstructed_cp)
Cl = compute_cl(xpos, ypos, zpos, reconstructed_cp, S)
computeError(Cl_5, Cl)   
    

plt.show()
