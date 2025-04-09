import numpy as np
import matplotlib.pyplot as pp
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
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

# Function to separate the Cp values above the airfoil from the ones below the airfoil
def separate_Cp(cp1, x1, y1, show, text):

    #Creation of the lists where the values will be saved
    CPlower = []
    xlower = []
    CPupper = []
    xupper = []

    #We loop through all the Cp values
    i=0
    while i < len(cp1):
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

        i += 1

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

#Function to reduce the amount of data. We only need 80% of the data to train the model.
#This function allows us to take the values that correspond to the indices established
def reduce_data(data, indices, axis):
    
    return data.take(indices, axis=axis)

# Function to truncate the decimals
def truncate(arr, decimals = 5):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor

# Function to calculate the error in the lift coefficient and in the aerodynamic centre
def computeError(ACL, OGACL):

    print("-----CL-----")

    newCL = ACL[0]
    ogCL = OGACL[0]

    #Calculation of the absolute error and truncation of the result
    CL_abs_error = truncate(np.abs(ogCL-newCL))
    #Calculation of the relative error and truncation of the result
    CL_rel_error = truncate(CL_abs_error/ogCL*100)

    print(str(CL_abs_error))
    print(str(CL_rel_error), "%")

    print("-----AC-----")

    newAC = ACL[1]
    ogAC = OGACL[1]

    #Same calculation as the one made for the lift coefficient
    AC_abs_error = truncate(np.abs(ogAC-newAC))
    AC_rel_error = truncate(AC_abs_error/ogAC*100)

    print(str(AC_abs_error))
    print(str(AC_rel_error), "%")

    #We return an array with the error values
    return np.array([[CL_abs_error, CL_rel_error], [AC_abs_error, AC_rel_error]])

#Function that performs the interpolation
def predict(ACL, OGACL):

    newCP = rom.predict(ACL).snapshots_matrix # interpolated values
    newCP = newCP.T
    print(newCP.shape)
    #We load the file containing the Cp values from the simulation of the corresponding alfa and mach
    name = 'C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\' + f"Cp_Alfa_{ACL[0]}_Mach_{ACL[1]}.txt"
    Cp_real = read_filename(name)
    #Calculation of the Cp errors and the simulation error
    Cp_error_abs, Cp_error_rel = compute_Cp_error(Cp_real, newCP)
    #We plot the Cp_real vs the Cp_interpolated to see how the interpolation varies from the real one
    text = "A = " + str(ACL[0]) + " M = " + str(ACL[1])
    plot_cp(Cp_real, newCP, text)
    t = "New: A = " + str(ACL[0]) + " M = " + str(ACL[1])
    #Calculation of the lift and of the aerodynamic centre
    Cl = compute_CL(newCP, xpos, ypos, True, t)
    AC = compute_AC(newCP, xpos, ypos, False, t)
    Cl = truncate(Cl)
    AC = truncate(AC)
    print(" CL  = ", Cl, " AC  = ", AC)
    print("OGCL = ", OGACL[0], "OGAC = ", OGACL[1])

    computeError([Cl, AC], OGACL)

    return

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
    rel_error = (abs_error/np.abs(Cp_real))*100
    
    #Calculation of the simulation error by obtaining the module of the relative error's vector, and dividing it by the module of the real Cp.
    sim_error = np.linalg.norm(Cp_real-Cp_interp)/np.linalg.norm(Cp_real)
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
    pp.title('Comparación de Cp real vs Cp interpolado ' + text )
    pp.legend()  # Mostrar leyenda
    pp.grid(True)

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


    
trainCount = int(np.floor(2000*0.8))
epsilon = 95
rank = 50
kernel = 'gaussian'
AlphaRange = [0, 2]
MachRange = [0.6, 0.75]

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

#We save in a database the input parameters (Alpha and Mach) and the output values(Cp)
db = Database(parameters, Cp.T)
#In this case SVD is not used, which means that the rank is the whole matrix of data.
pod = POD('svd', rank = Cp.shape[1])
rbf = RBF(kernel = kernel, epsilon = epsilon)  # radial basis function interpolator instance

#Combination of all the model
rom = ROM(db, pod, rbf)
#It adjust the data
rom.fit()

#Prediction of the Cp with the testing samples
print("--------------T1--------------")
start_time1 = time.perf_counter()
CPT1 = predict(T1, T1OG)
end_time1 = time.perf_counter()
elapsed_time1 = end_time1 - start_time1
print("Tiempo de ejecucion: " + str(elapsed_time1))
print("--------------T2--------------")
start_time2 = time.perf_counter()
CPT2 = predict(T2, T2OG)
end_time2 = time.perf_counter()
elapsed_time2 = end_time2 - start_time2
print("Tiempo de ejecucion: " + str(elapsed_time2))
print("--------------T3--------------")
start_time3 = time.perf_counter()
CPT3 = predict(T3, T3OG)
end_time3 = time.perf_counter()
elapsed_time3 = end_time3 - start_time3
print("Tiempo de ejecucion: " + str(elapsed_time3))
print("--------------T4--------------")
start_time4 = time.perf_counter()
CPT4 = predict(T4, T4OG)
end_time4 = time.perf_counter()
elapsed_time4 = end_time4 - start_time4
print("Tiempo de ejecucion: " + str(elapsed_time4))
print("--------------T5--------------")
start_time5 = time.perf_counter()
CPT5 = predict(T5, T5OG)
end_time5 = time.perf_counter()
elapsed_time5 = end_time5 - start_time5
print("Tiempo de ejecucion: " + str(elapsed_time5))


pp.show()