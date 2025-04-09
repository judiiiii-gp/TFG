import numpy as np
import matplotlib.pyplot as pp
import os
import time

# Function to calculate the lift coefficient
def compute_CL(cp, x_c, y_y, show, text):

    #First we need to separate the values of Cp that correspond to the top of the airfoil from the ones that correspond to the bottom
    #In this case we will get a matrix that in column 0 it will have the X values and in column 1 the Cp values corresponding to that X
    CP_lower, CP_upper = separate_Cp(cp, x_c, y_y, show, text)

    #The X values only go from 0 to 1, so here we establish that if some value is greater then 1 it will be eliminated
    CP_lower = CP_lower[CP_lower[:, 0] < 1]
    CP_upper = CP_upper[CP_upper[:, 0] < 1]

    #Division between the x_values and the Cp_values into different vectors
    x_lower, cp_lower = CP_lower[:, 0], CP_lower[:, 1]
    x_upper, cp_upper = CP_upper[:, 0], CP_upper[:, 1]

    #Performance of the integral of Cp over the X values to find the lift coefficient in the top and in the bottom
    LOWER = np.trapezoid(cp_lower, x_lower)
    UPPER = np.trapezoid(cp_upper, x_upper)

    #The total lift coefficient is found
    lift = LOWER - UPPER

    return lift

# Function to calculate the aerodynamic centre
def compute_AC(cp, x_c, y_y, show, text):

    #First we separate the Cp values from the top of the airfoil from the ones at the bottom of it
    CP_lower, CP_upper = separate_Cp(cp, x_c, y_y, show, text)

    #We perform the same as in the compute_CL function
    CP_lower = CP_lower[CP_lower[:, 0] < 1]
    CP_upper = CP_upper[CP_upper[:, 0] < 1]

    x_lower, cp_lower = CP_lower[:, 0], CP_lower[:, 1]
    x_upper, cp_upper = CP_upper[:, 0], CP_upper[:, 1]

    lift_lower = np.trapezoid(cp_lower, x_lower)
    lift_upper = np.trapezoid(cp_upper, x_upper)
    lift = lift_lower - lift_upper

    #To find the moment coefficient we need to integrate the product between Cp*X
    CM_le_lower = np.trapezoid(cp_lower * x_lower, x_lower)
    CM_le_upper = np.trapezoid(cp_upper * x_upper, x_upper)
    CM_le = CM_le_lower - CM_le_upper

    #To find the aerodynamic centre we divide the moment coefficient by the lift coefficient
    x_ac = CM_le / lift

    return x_ac

# # Function to separate the Cp values above the airfoil from the ones below the airfoil
def separate_Cp(cp1, x1, y1, show, text):

    #Creation of the lists where the values will be saved
    CPlower = []
    xlower = []
    CPupper = []
    xupper = []

    #We loop through all the Cp values
    for i in range(len(cp1)):
        #The value of x cannot be greater than 1, so if it's greater the point won't be added
        if x1[i] < 1:
            #y = 0 means the center of the airfoil, so if y<0 it means that it corresponds to the values of the bottom of the airfoil
            if y1[i] < 0:
                CPlower.append(cp1[i])
                xlower.append(x1[i])
            #If y is positive, the values will correspond to the top of the airfoil
            else:
                CPupper.append(cp1[i])
                xupper.append(x1[i])

    #Conversion of the lists into arrays
    CPlower = np.array(CPlower)
    CPupper = np.array(CPupper)
    xlower = np.array(xlower)
    xupper = np.array(xupper)

    #In the plots the pressure distribution is separated between the top one and the bottom one
    if show:

        pp.figure()
        pp.plot(xupper, CPupper, label="Upper Surface", marker="o")
        pp.plot(xlower, CPlower, label="Lower Surface", marker="x")
        pp.gca().invert_yaxis()  # Inversion of the y axis, so the negative values are shown in the top part
        pp.title("Pressure Coefficient (Cp) Distribution " + text)
        pp.xlabel("x")
        pp.ylabel("Cp")
        pp.legend()
        pp.grid()
        pp.show(block=False)

    #We create an array for the lower values and an array for the upper values
    lower_array = np.column_stack((xlower, CPlower))
    upper_array = np.column_stack((xupper, CPupper))

    return lower_array, upper_array

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

# Function to read a file with the Cp values from the simulation
def read_filename(filename):
    try:
        #We load the values inside the file and save them in Cp_values. The values in the file are separated by ';'
        Cp_values = np.loadtxt(filename, delimiter=";")
        return Cp_values
    except Exception as e:
        print(f"Error al leer el fichero {filename} : {e}")
        return None

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

# Function to make a plot to compare the Cp from the simulation and the interpolated Cp
def plot_cp(Cp_real, Cp_interpolado, text):
    pp.figure()
    pp.scatter(Cp_real, Cp_interpolado, color='blue', label='Cp interpolado vs Cp real')
    pp.plot([min(Cp_real), max(Cp_real)], [min(Cp_real), max(Cp_real)], color='red', linestyle='--', label='Interpolación perfecta (y = x)')

    pp.xlabel('Cp real')
    pp.ylabel('Cp interpolado')
    pp.title('Comparación de Cp real vs Cp interpolado ' + text)
    pp.legend()
    pp.grid(True)

#Function to reduce the amount of data. We only need 80% of the data to train the model.
#This function allows us to take the values that correspond to the indices established
def reduce_data(data, indices, axis):
    return data.take(indices, axis=axis)

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

#Function where we perform calculations explained before in the functions
def calculations(Cp_interpolated, A_M, Cl_Ac, X, Y):
    
    #We load the file containing the Cp values from the simulation of the corresponding alfa and mach
    name = 'C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\' + f"Cp_Alfa_{A_M[0,0]}_Mach_{A_M[0,1]}.txt"
    Cp_real = read_filename(name)
    #Calculation of the Cp errors and the simulation error
    Cp_error_abs, Cp_error_rel = compute_Cp_error(Cp_real, Cp_interpolated)
    #We plot the Cp_real vs the Cp_interpolated to see how the interpolation varies from the real one
    text = "A = " + str(A_M[0,0]) + " M = " + str(A_M[0,1])
    plot_cp(Cp_real, Cp_interpolated, text)
    # hist_plot(Cp_error_abs, text)
    t = "New: A = " + str(A_M[0,0]) + " M = " + str(A_M[0,1])
    #Calculation of the lift and of the aerodynamic centre
    Cl = compute_CL(Cp_interpolated, X, Y, True, t)
    AC = compute_AC(Cp_interpolated, X, Y, False, t)
    
    Cl = truncate(Cl)
    AC = truncate(AC)
    print(" CL  = ", Cl, " AC  = ", AC)
    print("OGCL = ", Cl_Ac[0], "OGAC = ", Cl_Ac[1])

    computeError([Cl, AC], Cl_Ac)

#Main function
def main():
    data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Exemple TFG\\CODE\\Naca0012_database_mesh_1\\FOM_Skin_Data"

    rank = 50
    epsilon = 95
    kernel = 'gaussian'
    AlphaRange = [0, 2]
    MachRange = [0.6, 0.75]
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
    
    #We load the data from the database
    Alpha, Mach, Cp, xpos, ypos = load_data(data_path)
    
    #We truncate the Alpha and Mach values to 5 decimals
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
    trainCount = int(np.floor(parameters.shape[0] * 0.8))

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
    calculations(reconstructed_cp.T, T1, T1OG, xpos, ypos)
    
    print("------ TEST 2 ------")
    start_time2 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T2, epsilon, kernel)
    end_time2 = time.perf_counter()
    elapsed_time2 = end_time2 - start_time2
    print("Tiempo de ejecucion: " + str(elapsed_time2))
    reconstructed_cp = interpolated_coefficients @ U_r.T
    calculations(reconstructed_cp.T, T2, T2OG, xpos, ypos)
    
    print("------ TEST 3 ------")
    start_time3 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T3, epsilon, kernel)
    end_time3 = time.perf_counter()
    elapsed_time3 = end_time3 - start_time3
    print("Tiempo de ejecucion: " + str(elapsed_time3))
    reconstructed_cp = interpolated_coefficients @ U_r.T
    calculations(reconstructed_cp.T, T3, T3OG, xpos, ypos)
    
    print("------ TEST 4 ------")
    start_time4 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T4, epsilon, kernel)
    end_time4 = time.perf_counter()
    elapsed_time4 = end_time4 - start_time4
    print("Tiempo de ejecucion: " + str(elapsed_time4))
    reconstructed_cp = interpolated_coefficients @ U_r.T
    calculations(reconstructed_cp.T, T4, T4OG, xpos, ypos)
    
    print("------ TEST 5 ------")
    start_time5 = time.perf_counter()
    interpolated_coefficients = rbf_interpolate(parameters, weights, T5, epsilon, kernel)
    end_time5 = time.perf_counter()
    elapsed_time5 = end_time5 - start_time5
    print("Tiempo de ejecucion: " + str(elapsed_time5))
    reconstructed_cp = interpolated_coefficients @ U_r.T
    calculations(reconstructed_cp.T, T5, T5OG, xpos, ypos)
    
if __name__ == '__main__':
    main()
    pp.show()
