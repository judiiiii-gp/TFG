import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.spatial import Delaunay, ConvexHull

# Function to truncate the decimals
def truncate(arr, decimals=5):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor

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

#Function that performs the randomized SVD
def svd_pod(data, rank):
    
    #We find the shape of our data array
    m, n = data.shape

    #Generation of a random array with n rows and k (the rank chosen) columns
    P = np.random.randn(n, rank)
    #Creation of the Z matrix, which captures the most significant features of the data matrix
    Z = data @ P
    #QR factorization of the Z matrix
    Q, _ = np.linalg.qr(Z)
    #Projection of the data matrix into the basis Q
    Y = Q.T @ data
    #Performance of the SVD of the reduced matrix Y
    U_tilde, Sigma, Vt = np.linalg.svd(Y, full_matrices=False)
    #We return U to the original column space
    U = Q @ U_tilde

    return U, Sigma, Vt

#Function that implements the RBF function
def rbf_kernel(x, c, epsilon, kernel):
    #Calculation fo the Euclidean distance between the points
    r = np.sqrt(np.sum((x - c) ** 2, axis=1))
    #If we choose the gaussian function
    if kernel == 'gaussian':
        return np.exp(-(epsilon*r)**2)
    #If we choose the linear function
    elif kernel == 'linear':
        return r
    else:
        raise ValueError("Unsupported kernel type")

#Function to compute the weights of the system
def compute_rbf_weights(X, Y, epsilon, kernel):
    #Creation of a matrix that will have all the results from the RBF function
    N = X.shape[0]
    A = np.zeros((N, N))
    #We loop through all X values, to perform the RBF function with all the combinations in X. The results from all the combinations are stored in A.
    for i in range(N):
        A[i, :] = rbf_kernel(X[i], X, epsilon, kernel)
    
    #We solve the linear system to find the weights
    weights = np.linalg.solve(A, Y)

    return weights

#Function that performs the interpolation
def rbf_interpolate(X_train, weights, X_new, epsilon, kernel):
    N_new = X_new.shape[0]  # Number of new points
    N = X_train.shape[0]    # Number of training points
    M = weights.shape[1]    # Number of modes
    interpolated = np.zeros((N_new, M))  # We create the interpolated matrix, where the interpolated values will be stored
    
    #We loop through all the new points
    for i in range(N_new):
        #We perform the RBF function ith all the combinations between the new points and the training points. 
        phi = rbf_kernel(X_new[i], X_train, epsilon, kernel)

        #By performing the dot product between the vector of the results obtained from the RBF function and the vector of the system's weights
        #we find the interpolated value, which is saved in the interpolated matrix
        interpolated[i, :] = np.dot(phi, weights)
    
    return interpolated

def plot_cp_3d(xpos, ypos, zpos, Cp, Alpha, Mach):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xpos, ypos, zpos, c=Cp, cmap='viridis')
    fig.colorbar(sc, ax=ax, label='Cp')
    ax.set_title(f'Cp distribution (Alpha={Alpha}, Mach={Mach})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
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


def main():
    
    data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\BBDD 3D\\FOM_Skin_Data"
    rank = 50
    epsilon = 95
    kernel = 'linear'
    AlphaRange = [0, 3.5]
    MachRange = [0.6, 0.85]
    
    T1 = np.array([[0.15232, 0.60535]])
    T2 = np.array([[1.0615, 0.62832]])
    T3 = np.array([[2.15525, 0.73018]])
    T4 = np.array([[2.89353, 0.79088]])
    T5 = np.array([[3.44724, 0.80288]])
    
    Alpha, Mach, Cp, xpos, ypos, zpos = load_data(data_path)
    
    Alpha = truncate(Alpha)
    Mach = truncate(Mach)
    
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
    
    #We apply the SVD to the matrix Cp
    U_r, S_r, Vt_r = svd_pod(Cp, rank)
    #We obtain the coefficients from the SVD (the reduced Cp matrix)
    coefficients = (Vt_r.T * S_r)
    #Computation of the weights of the system
    weights = compute_rbf_weights(parameters, coefficients, epsilon, kernel)
    
    #Testing phase, where we perform the interpolation and the calculation fo the errors for each sample.
    print("------ TEST 1 ------")
    start_time1 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T1, epsilon, kernel)
    end_time1 = time.perf_counter()
    elapsed_time1 = end_time1 - start_time1
    print("Tiempo de ejecucion: " + str(elapsed_time1))
    #We return the interpolated values to the original column space
    reconstructed_cp = interpolated_coefficients @ U_r.T
    coords = np.column_stack((xpos, ypos, zpos))
    print(reconstructed_cp)
    Cl = calcular_cl_desde_puntos(coords, reconstructed_cp.T, T1[0, 0], T1[0, 1])
    print("C_L =", Cl)
    # plot_cp_3d(xpos, ypos, zpos, reconstructed_cp, T1[0, 0], T1[0, 1]) 
    
    print("------ TEST 2 ------")
    start_time2 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T2, epsilon, kernel)
    end_time2 = time.perf_counter()
    elapsed_time2 = end_time2 - start_time2
    print("Tiempo de ejecucion: " + str(elapsed_time2))
    #We return the interpolated values to the original column space
    reconstructed_cp = interpolated_coefficients @ U_r.T
    coords = np.column_stack((xpos, ypos, zpos))
    print(reconstructed_cp)
    Cl = calcular_cl_desde_puntos(coords, reconstructed_cp.T, T2[0, 0], T2[0, 1])
    print("C_L =", Cl)
    
    print("------ TEST 3 ------")
    start_time3 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T3, epsilon, kernel)
    end_time3 = time.perf_counter()
    elapsed_time3 = end_time3 - start_time3
    print("Tiempo de ejecucion: " + str(elapsed_time3))
    #We return the interpolated values to the original column space
    reconstructed_cp = interpolated_coefficients @ U_r.T
    coords = np.column_stack((xpos, ypos, zpos))
    print(reconstructed_cp)
    Cl = calcular_cl_desde_puntos(coords, reconstructed_cp.T, T3[0, 0], T3[0, 1])
    print("C_L =", Cl)
    
    print("------ TEST 4 ------")
    start_time4 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T4, epsilon, kernel)
    end_time4 = time.perf_counter()
    elapsed_time4 = end_time4 - start_time4
    print("Tiempo de ejecucion: " + str(elapsed_time4))
    #We return the interpolated values to the original column space
    reconstructed_cp = interpolated_coefficients @ U_r.T
    coords = np.column_stack((xpos, ypos, zpos))
    print(reconstructed_cp)
    Cl = calcular_cl_desde_puntos(coords, reconstructed_cp.T, T4[0, 0], T4[0, 1])
    print("C_L =", Cl)
    
    print("------ TEST 5 ------")
    start_time5 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T5, epsilon, kernel)
    end_time5 = time.perf_counter()
    elapsed_time5 = end_time5 - start_time5
    print("Tiempo de ejecucion: " + str(elapsed_time5))
    #We return the interpolated values to the original column space
    reconstructed_cp = interpolated_coefficients @ U_r.T
    coords = np.column_stack((xpos, ypos, zpos))
    print(reconstructed_cp)
    Cl = calcular_cl_desde_puntos(coords, reconstructed_cp.T, T5[0, 0], T5[0, 1])
    print("C_L =", Cl)

if __name__ == '__main__':
    main()
    plt.show()
