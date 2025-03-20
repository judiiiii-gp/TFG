import numpy as np
import matplotlib.pyplot as pp
from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
import os

#<>

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

    CPlower = []
    xlower = []
    CPupper = []
    xupper = []

    i=0
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
        pp.gca().invert_yaxis()  # Invert Y-axis to match aerodynamic convention
        pp.title("Pressure Coefficient (Cp) Distribution " + text)
        pp.xlabel("x")
        pp.ylabel("Cp")
        pp.legend()
        pp.grid()
        pp.show(block=False)

    lower_array = np.column_stack((xlower, CPlower))
    upper_array = np.column_stack((xupper, CPupper))
    
    return lower_array, upper_array

def reduce_data(data, indices, axis):
    
    return data.take(indices, axis=axis)

def truncate(arr, decimals = 5):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor

def computeError(ACL, OGACL):

    print("-----CL-----")

    newCL = ACL[0]
    ogCL = OGACL[0]

    CL_abs_error = truncate(np.abs(ogCL-newCL))
    CL_rel_error = truncate(CL_abs_error/ogCL*100)

    print(str(CL_abs_error))
    print(str(CL_rel_error), "%")

    print("-----AC-----")

    newAC = ACL[1]
    ogAC = OGACL[1]

    AC_abs_error = truncate(np.abs(ogAC-newAC))
    AC_rel_error = truncate(AC_abs_error/ogAC*100)

    print(str(AC_abs_error))
    print(str(AC_rel_error), "%")

    return np.array([[CL_abs_error, CL_rel_error], [AC_abs_error, AC_rel_error]])

def predict(ACL, OGACL):

    newCP = rom.predict(ACL).snapshots_matrix # interpolated values
    newCP = newCP.T
    name = 'C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\' + f"Cp_Alfa_{ACL[0]}_Mach_{ACL[1]}.txt"
    Cp_real = read_filename(name)
    Cp_error_abs, Cp_error_rel = compute_Cp_error(Cp_real, newCP)
    text = "A = " + str(ACL[0]) + " M = " + str(ACL[1])
    plot_cp(Cp_real, newCP, text)
    # hist_plot(Cp_error_abs, text)
    t = "New: A = " + str(ACL[0]) + " M = " + str(ACL[1])
    Cl = compute_CL(newCP, xpos, ypos, True, t)
    AC = compute_AC(newCP, xpos, ypos, False, t)
    Cl = truncate(Cl)
    AC = truncate(AC)
    print(" CL  = ", Cl, " AC  = ", AC)
    print("OGCL = ", OGACL[0], "OGAC = ", OGACL[1])

    computeError([Cl, AC], OGACL)

    return

def read_filename(filename):
    try:
        Cp_values = np.loadtxt(filename, delimiter=";")
        return Cp_values
    except Exception as e:
        print(f"Error al leer el fichero {filename} : {e}")
        return None

def compute_Cp_error(Cp_real, Cp_interp):
    abs_error = np.abs(Cp_real - Cp_interp)
    
    rel_error = (abs_error/np.abs(Cp_real))*100
    
    sim_error = np.linalg.norm(Cp_real-Cp_interp)/np.linalg.norm(Cp_real)
    sim_error = truncate(sim_error)
    print("Error global de la simulación: " + str(sim_error))
    return abs_error, rel_error

def plot_cp(Cp_real, Cp_interpolado, text):
    pp.figure()
    pp.scatter(Cp_real, Cp_interpolado, color='blue', label='Cp interpolado vs Cp real')
    pp.plot([min(Cp_real), max(Cp_real)], [min(Cp_real), max(Cp_real)], color='red', linestyle='--', label='Interpolación perfecta (y = x)')
    
    pp.xlabel('Cp real')
    pp.ylabel('Cp interpolado')
    pp.title('Comparación de Cp real vs Cp interpolado ' + text )
    pp.legend()  # Mostrar leyenda
    pp.grid(True)

# def heat_map(X, Cp_error_rel):
#     pp.figure()  # Tamaño de la figura
#     heatmap = pp.scatter(X, np.zeros_like(X), c=Cp_error_rel, cmap='viridis', s=100)
#     cbar = pp.colorbar(heatmap)
#     cbar.set_label('Error absoluto')  
#     pp.xlabel('Posición a lo largo del perfil')
#     pp.yticks([])  # Ocultar el eje Y (no es necesario en este caso)
#     pp.title('Mapa de calor del error absoluto en el perfil')
#     pp.grid(True, linestyle='--', alpha=0.5)

# def hist_plot(Cp_error, text):
#     pp.figure()
#     pp.hist(Cp_error, bins=20, color='blue', edgecolor='black', alpha=0.7)
#     pp.xlabel('Error absoluto')
#     pp.ylabel('Frecuencia')
#     pp.title('Histograma del error absoluto ' + text)
#     pp.grid(True, linestyle='--', alpha=0.5)

#SCRIPT

# TRAINING DATA
# ALPHA 0-2
# MACH 0.6-0.75
trainCount = int(np.floor(2000*0.8))
epsilon = 95
rank = 50
kernel = 'gaussian'

AlphaRange = [0, 2]
MachRange = [0.6, 0.75]

VL = [1, 0.72]
VLOG = [0.19327, 0.25]

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

T6 = [1, 0.72]
T6OG = [1, 1]


Data_List = os. listdir("C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\Naca0012_database_mesh_1\\FOM_Skin_Data")

CP = np.zeros(1)
Alpha = np.zeros((len(Data_List), 1))
Mach = np.zeros((len(Data_List), 1))
xpos = np.array([])
xpos = np.array([])

i = 0
while i < len(Data_List):
    
    param = Data_List[i].removesuffix('.dat')
    paramMatrix = param.split(',')

    dir = 'C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\Naca0012_database_mesh_1\\FOM_Skin_Data\\' + Data_List[i]

    file1 = open(dir, 'r')
    lines = file1.readlines()

    if i == 0:
        CP = np.zeros(len(lines))
        xpos = np.zeros(len(lines))
        ypos = np.zeros(len(lines))

    Alpha[i] = paramMatrix[0]
    Mach[i] = paramMatrix[1]

    if i == 0:
        A = np.zeros((len(lines), len(Data_List))) 

    j=0
    while j < len(lines):
        line = lines[j]
        lineMatrix = line.split(' ')
        lineMatrix[3] = lineMatrix[3].removesuffix('\n')

        CP[j] = lineMatrix[3] #CP
        xpos[j] = lineMatrix[0] #X
        ypos[j] = lineMatrix[1] #Y

        j += 1
    
    k = 0
    while k < len(CP):
        A[k, i] = CP[k]
        k +=1

    i += 1


Alpha = truncate(Alpha)
Mach = truncate(Mach)

TestSamples = np.stack((T1, T2, T3, T4, T5))

i = 0
removeMatrix = np.array([])
while i < len(Alpha):
    if AlphaRange[0] < Alpha[i] < AlphaRange[1]:
        if MachRange[0] < Mach[i] < MachRange[1]:
            kk = 0
        else:
            removeMatrix = np.append(removeMatrix, i)
    else:
        removeMatrix = np.append(removeMatrix, i)
    
    y = 0
    while y < TestSamples.shape[0]:
        if Alpha[i] == TestSamples[y, 0] and Mach[i] == TestSamples[y, 1]:
            removeMatrix = np.append(removeMatrix, i)
        y+=1

    i+=1


removeMatrix = removeMatrix.astype(int)

Alpha = np.delete(Alpha, removeMatrix, axis=0)
Mach = np.delete(Mach, removeMatrix, axis=0)
A = np.delete(A, removeMatrix, axis=1)

#RBF 

if(len(Alpha) > trainCount):
    np.random.seed(42)
    selected_indices = np.random.choice(len(Alpha), trainCount, replace=False)

    Alpha = reduce_data(Alpha, selected_indices, 0)
    Mach = reduce_data(Mach, selected_indices, 0)
    A = reduce_data(A, selected_indices, 1)

print("Samples: ", len(Alpha))

parameters = np.column_stack((Alpha, Mach))


db = Database(parameters, A.T)
pod = POD('svd', rank = A.shape[1])
rbf = RBF(kernel = kernel, epsilon = epsilon)  # radial basis function interpolator instance

rom = ROM(db, pod, rbf)
rom.fit()

# print("----------VALIDATION----------")
# CPVL = predict(VL, VLOG)

# print("Execution started...")
# input("Press Enter to continue...")
# print("Execution resumed!")

print("--------------T1--------------")
CPT1 = predict(T1, T1OG)
print("--------------T2--------------")
CPT2 = predict(T2, T2OG)
print("--------------T3--------------")
CPT3 = predict(T3, T3OG)
print("--------------T4--------------")
CPT4 = predict(T4, T4OG)
print("--------------T5--------------")
CPT5 = predict(T5, T5OG)

pp.show()