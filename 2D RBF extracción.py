import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###########################################################################
###########################################################################
################ FUNCTIONS ###############################################z
    
#Function that calculates the weights of the RBF
def compute_rbf_weights(X, Y, epsilon):

    #First we calculate the Euclidean distance between all the points in the matrix
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis = -1)
    
    #Now we apply the RBF function
    K = rbf(type, epsilon, dist_matrix)
    
    #Finally we can compute the weights by solving a linear system
    W = np.linalg.solve(K, Y)

    return W

#RBF function
def rbf(type, epsilon, distance):
    #If type == 0, the linear RBF function is used
    if type ==0:
        result = distance
    #If type =! 0, the gaussian RBF function is used
    else:
        result = np.exp(-1*distance*distance*epsilon*epsilon)
    return result

#Function that interpolates the Cp values thanks to the weights calculated before
def interpolation(X_known, W, X_new, epsilon):
    
    #We calculate the Euclidean distance between the new point and all the known points
    dist_new = np.array([np.sqrt((X_new[0]-p[0])**2 + (X_new[1]-p[1])**2)for p in X_known])
    
    #We perform the RBF function with the distance calculated
    K_new = rbf(type, epsilon, dist_new)
    
    #With the weights we are able to interpolate the new values of the Cp
    result = np.dot(K_new, W)
    return result

#Main function of the interpolation
def predict(parameters_test):
    #We interpolate using the RBF method thanks to the weights previously calculated
    newCp = interpolation(parameters_train, W, parameters_test, epsilon)
    newCp = newCp.T
    
    text2 = "New Attack angle = " + str(parameters_test[0]) + " Mach = " + str(parameters_test[1])
    #With the calculated pressure distribution we can calculate the Cl and the Ac
    Cl = calc_Cl(X, Y, newCp, False, "Simulated ", text2)
    Ac = calc_Ac(X, Y, newCp, False, "Simulated ",text2)
    
    return Cl, Ac

#Function to separate the values of Cp and X that correspond to the upper surface from the ones that correspond to the lower surface
def sep_Cp(X, Y, Cp, show, text1, text2):
    #Lists where the values will be stored
    Cp_top = []
    X_top =[]
    Cp_bottom = []
    X_bottom = []
    i = 0
    while i<len(Cp):
        #We only want the values of x that go from 0 to 1
        if(X[i]<1):
            #Values that correspond to the lower surface
            if (Y[i]<0):
                Cp_bottom.append(Cp[i])
                X_bottom.append(X[i])
            #Values that correspond to the upper surface
            else:
                Cp_top.append(Cp[i])
                X_top.append(X[i])
        i+=1
    #We convert the lists into arrays
    Cp_top = np.array(Cp_top)
    X_top = np.array(X_top)
    Cp_bottom = np.array(Cp_bottom)
    X_bottom = np.array(X_bottom)
    
    #If it's true we will plot the pressure distribution
    if show:
        plot_Cp(X_bottom, Cp_bottom, X_top, Cp_top, text1, text2)
    
    return X_bottom, Cp_bottom, X_top, Cp_top

#Function to calculate the lift coefficient
def calc_Cl(X, Y, Cp, show, text1, text2):
    
    #We need to separate the Cp_top values from the bottom ones
    X_bottom, Cp_bottom, X_top, Cp_top = sep_Cp(X, Y, Cp, show, text1, text2)
    X_bottom = X_bottom[X_bottom< 1]
    X_top = X_top[X_top< 1]
    
    #The lift coefficient is found by integrating the difference between the Cp_low and Cp_top
    #Firs we perform the integration of Cp over the airfoil's surface separately
    Cl_top = np.trapezoid(Cp_top, X_top)
    Cl_bottom = np.trapezoid(Cp_bottom, X_bottom)
    
    #Once the integration is complete, we can perform the difference between the lower and upper value
    Cl = Cl_bottom - Cl_top

    return Cl

#Function to calculate the aerodynamic centre
def calc_Ac(X, Y, Cp, show, text1, text2):
    #We need to separate Cp_top from Cp_bottom
    X_bottom, Cp_bottom, X_top, Cp_top = sep_Cp(X, Y, Cp, show, text1, text2)
    X_bottom = X_bottom[X_bottom< 1]
    X_top = X_top[X_top< 1]
    
    #To find the aerodynamic center firs we need to find Cm
    #To find Cm we will need to divide the integral of Cp*X by the integral of x
    #This will be done separately for both surfaces and then the global CM will be found 
    CM_top = np.trapezoid(Cp_top*X_top, X_top)
    CM_bottom = np.trapezoid(Cp_bottom*X_bottom, X_bottom)
    #Global CM
    CM = CM_bottom - CM_top
    
    #Then we will need to compute the lift, because the aerodynamic centre is found by dividing CM by Cl
    Cl_top = np.trapezoid(Cp_top, X_top)
    Cl_bottom = np.trapezoid(Cp_bottom, X_bottom)
    Cl = Cl_bottom - Cl_top

    #Aerodynamic centre
    x_ac = CM/Cl

    return x_ac

#Function to calculate the error in the interpolation
def err_calc(real, interpolated):
    
    #Calculation of the absolute error
    error_abs = np.abs(real-interpolated)

    
    #Calculation of the relative error
    error_rel = np.where(real != 0, (error_abs / np.abs(real))*100, 0)

    
    return error_abs, error_rel

#Function to plot the pressure distribution around the airfoil
def plot_Cp(X_bottom, Cp_bottom, X_top, Cp_top, text1, text2):
    plt.figure()
    plt.plot(X_bottom, Cp_bottom, label = "Lower surface", marker = 'o')
    plt.plot(X_top, Cp_top, label = "Upper surface", marker = 'x')
    plt.title(text1 + " Pressure distribution surface " + text2)
    plt.xlabel("X")
    plt.ylabel("Cp")
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()
    plt.show(block=False)

#We reduce the data 
def reduce(matrix, indices, axis):

    return matrix.take(indices, axis=axis)

#Function to reduce the number of decimals on the values. 
def decimals(A, N):
    factor = 10.0 ** N
    return np.floor(A * factor)/factor


######################################################################
######################################################################
######## START OF THE SCRIPT #########################################

#Variables
por=80
trainCount = int(np.floor(2000*por/100))
epsilon = 95
rank = 50
type = 0

#Training data
Attack_angle_range = [0, 2]
Mach_range = [0.6, 0.75]

VL = [1, 0.72]
VLOG = [0.19327, 0.25]

T1 = [0.12537, 0.69466]
T1OG = [0, 0]
T2 = [1.46424, 0.65413]
T2OG = [0, 0]
T3 = [0.80018, 0.68952]
T3OG = [0, 0]
T4 = [1.21033, 0.71942]
T4OG = [0, 0]
T5 = [1.98963, 0.71921]
T5OG = [0, 0]

T6 = [1, 0.72]
T6OG = [1, 1]

dir = 'C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Exemple TFG\\CODE\\Naca0012_database_mesh_1\\FOM_Skin_Data'
#The list of files in the directory is converted into an array 
list_dir = np.array(os.listdir(dir)) 


#Now we are going to extract the values of X, Y and Cp that are contained in the files
#We need to create the matrices to store the angle of attack and mach values of each file
Attack_angle = np.zeros((len(list_dir), 1))
Mach = np.zeros((len(list_dir),1))
i = 0
while i< len(list_dir):
    name = list_dir[i].replace(".dat", "")
    values = name.split(',')
    #We store the values in the corresponding index
    Attack_angle[i] = values[0]
    Mach[i] = values[1]
    complete_link = os.path.join(dir, list_dir[i])
    #Now we open the file to read it line by line to extract the values of X, Y and CP
    file = open(complete_link, 'r')
    lines = file.readlines()
    #We create the X and Y matrices.
    #Each file has the same values of X and Y, so we will only need to store them once. We won't need to store them differently for each file
    #The Cp value is different in each fle, so we will need to store all the values. 
    #The values of Cp will be stored as follows: for each file, the values of the Cp will be stored in a column. Which means that each column will contain the values of the Cp for a single file.
    X = np.zeros(len(lines))
    Y = np.zeros(len(lines))
    if (i==0):
        #We only create the matrix to store the Cp values in the first iteration
        Cp = np.zeros((len(lines), len(list_dir)))
    j = 0
    while j<len(lines):
        line = lines[j].split(' ')
        line[3] = line[3].removesuffix('\n')
        X[j] = line[0]
        Y[j] = line[1]
        Cp[j, i] = line[3]
        j+=1
        
    i+=1

TestSamples = np.vstack((T1, T2, T3, T4, T5))

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
    
    y = 0
    while y< TestSamples.shape[0]:
        if Attack_angle[i] == TestSamples[y, 0] and Mach[i] == TestSamples[y, 1]:
            removeMatrix = np.append(removeMatrix, i)
        y+=1
    i+=1

removeMatrix = removeMatrix.astype(int)

Attack_angle = np.delete(Attack_angle, removeMatrix, axis = 0)
Mach = np.delete(Mach, removeMatrix, axis=0)

Cp = np.delete(Cp, removeMatrix, axis = 1)

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



#Calculation of the weights of the system

W = compute_rbf_weights(parameters_train, Cp_train.T, epsilon)

resultados = []
t = 0
while t<len(parameters_test):
    Cl, Ac = predict(parameters_test[t])
    Cl = decimals(Cl, 5)
    Ac = decimals(Ac, 5)
    resultados.append([parameters_test[t, 0], parameters_test[t, 1], Cl, Ac])
    t+=1

df = pd.DataFrame(resultados, columns=["Alfa", "Mach", "Cl", "Ac"])
df.to_excel("resultados_interpolación_20250310.xlsx", index=False)

print("Done!")