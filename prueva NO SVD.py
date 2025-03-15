import numpy as np
import matplotlib.pyplot as pp
import os

# Función para calcular el coeficiente de sustentación (CL)
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

# Función para calcular el centro aerodinámico (AC)
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

# Función para separar los coeficientes de presión entre la superficie superior e inferior
def separate_Cp(cp1, x1, y1, show, text):

    CPlower = []
    xlower = []
    CPupper = []
    xupper = []

    for i in range(len(cp1)):

        if x1[i] < 1:
            if y1[i] < 0:
                CPlower.append(cp1[i])
                xlower.append(x1[i])
            else:
                CPupper.append(cp1[i])
                xupper.append(x1[i])

    CPlower = np.array(CPlower)
    CPupper = np.array(CPupper)
    xlower = np.array(xlower)
    xupper = np.array(xupper)

    if show:

        pp.figure()
        pp.plot(xupper, CPupper, label="Upper Surface", marker="o")
        pp.plot(xlower, CPlower, label="Lower Surface", marker="x")
        pp.gca().invert_yaxis()  # Invertir eje Y
        pp.title("Pressure Coefficient (Cp) Distribution " + text)
        pp.xlabel("x")
        pp.ylabel("Cp")
        pp.legend()
        pp.grid()
        pp.show(block=False)

    lower_array = np.column_stack((xlower, CPlower))
    upper_array = np.column_stack((xupper, CPupper))

    return lower_array, upper_array

# Función para truncar valores
def truncate(arr, decimals=5):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor

# Función para calcular errores entre coeficientes de sustentación y centro aerodinámico
def computeError(ACL, OGACL):

    print("-----CL-----")

    newCL = ACL[0]
    ogCL = OGACL[0]

    CL_abs_error = truncate(np.abs(ogCL - newCL))
    CL_rel_error = truncate((CL_abs_error / ogCL) * 100)

    print(str(CL_abs_error))
    print(str(CL_rel_error), "%")

    print("-----AC-----")

    newAC = ACL[1]
    ogAC = OGACL[1]

    AC_abs_error = truncate(np.abs(ogAC - newAC))
    AC_rel_error = truncate((AC_abs_error / ogAC) * 100)

    print(str(AC_abs_error))
    print(str(AC_rel_error), "%")

    return np.array([[CL_abs_error, CL_rel_error], [AC_abs_error, AC_rel_error]])

# Función para leer un archivo de coeficientes de presión
def read_filename(filename):
    try:
        Cp_values = np.loadtxt(filename, delimiter=";")
        return Cp_values
    except Exception as e:
        print(f"Error al leer el fichero {filename} : {e}")
        return None

# Función para calcular el error absoluto y relativo entre Cp reales e interpolados
def compute_Cp_error(Cp_real, Cp_interp):
    abs_error = np.abs(Cp_real - Cp_interp)
    rel_error = (abs_error / np.abs(Cp_real)) * 100

    sim_error = np.linalg.norm(Cp_real - Cp_interp) / np.linalg.norm(Cp_real)
    sim_error = truncate(sim_error)
    print("Error global de la simulación: " + str(sim_error))
    return abs_error, rel_error

# Función para graficar la comparación entre Cp real e interpolado
def plot_cp(Cp_real, Cp_interpolado, text):
    pp.figure()
    pp.scatter(Cp_real, Cp_interpolado, color='blue', label='Cp interpolado vs Cp real')
    pp.plot([min(Cp_real), max(Cp_real)], [min(Cp_real), max(Cp_real)], color='red', linestyle='--', label='Interpolación perfecta (y = x)')

    pp.xlabel('Cp real')
    pp.ylabel('Cp interpolado')
    pp.title('Comparación de Cp real vs Cp interpolado ' + text)
    pp.legend()
    pp.grid(True)

def truncate(arr, decimals=5):
    factor = 10.0 ** decimals
    return np.floor(arr * factor) / factor

def reduce_data(data, indices, axis):
    return data.take(indices, axis=axis)

def rbf_kernel(x, c, epsilon, kernel):
    r = np.linalg.norm(x - c, axis=1)
    if kernel == 'gaussian':
        return np.exp(-epsilon*epsilon*r*r)
    elif kernel == 'linear':
        return r
    else:
        raise ValueError("Unsupported kernel type")

def compute_rbf_weights(X, Y, epsilon, kernel):
    N = X.shape[0]
    A = np.zeros((N, N))
    for i in range(N):
        A[i, :] = rbf_kernel(X[i], X, epsilon, kernel)
    weights = np.linalg.solve(A, Y)

    # Si Y tiene la forma (N, M), entonces weights debe tener la misma forma
    if weights.ndim > 1 and weights.shape[1] != Y.shape[1]:
        weights = weights.T  # Ajustar la forma si es necesario para que coincidan las dimensiones
    
    return weights

def rbf_interpolate(X_train, weights, X_new, epsilon, kernel):
    N_new = X_new.shape[0]  # Número de puntos nuevos
    N = X_train.shape[0]    # Número de puntos de entrenamiento
    M = weights.shape[1]    # Número de modos, debe ser 50
    interpolated = np.zeros((N_new, M))  # La forma debe ser (N_new, M)
    
    for i in range(N_new):
        phi = rbf_kernel(X_new[i], X_train, epsilon, kernel)

        if phi.shape[0] != weights.shape[0]:
            raise ValueError(f"Mismatch in dimensions: phi={phi.shape}, weights={weights.shape}")

        interpolated[i, :] = np.dot(phi, weights)
    
    return interpolated

def load_data(data_path):
    data_list = os.listdir(data_path)
    num_files = len(data_list)

    sample_file = os.path.join(data_path, data_list[0])
    with open(sample_file, 'r') as f:
        num_points = len(f.readlines())

    Alpha = np.zeros((num_files, 1))
    Mach = np.zeros((num_files, 1))
    Cp = np.zeros((num_points, num_files))
    xpos = np.zeros(num_points)
    ypos = np.zeros(num_points)

    for i, file in enumerate(data_list):
        alpha, mach = map(float, file.replace('.dat', '').split(','))
        Alpha[i] = alpha
        Mach[i] = mach

        with open(os.path.join(data_path, file), 'r') as f:
            lines = f.readlines()
            for j, line in enumerate(lines):
                x, y, _, cp = map(float, line.split())
                Cp[j, i] = cp
                if i == 0:
                    xpos[j] = x
                    ypos[j] = y

    return Alpha, Mach, Cp, xpos, ypos

def calculations(Cp_interpolated, A_M, Cl_Ac, X, Y):
    name = 'C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Code\\' + f"Cp_Alfa_{A_M[0,0]}_Mach_{A_M[0,1]}.txt"
    Cp_real = read_filename(name)
    Cp_error_abs, Cp_error_rel = compute_Cp_error(Cp_real, Cp_interpolated)
    text = "A = " + str(A_M[0,0]) + " M = " + str(A_M[0,1])
    plot_cp(Cp_real, Cp_interpolated, text)
    # hist_plot(Cp_error_abs, text)
    t = "New: A = " + str(A_M[0,0]) + " M = " + str(A_M[0,1])
    Cl = compute_CL(Cp_interpolated, X, Y, True, t)
    AC = compute_AC(Cp_interpolated, X, Y, False, t)
    Cl = truncate(Cl)
    AC = truncate(AC)
    print(" CL  = ", Cl, " AC  = ", AC)
    print("OGCL = ", Cl_Ac[0], "OGAC = ", Cl_Ac[1])

    computeError([Cl, AC], Cl_Ac)

def main():
    data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Exemple TFG\\CODE\\Naca0012_database_mesh_1\\FOM_Skin_Data"

    epsilon = 95
    kernel = 'gaussian'
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
    
    Alpha, Mach, Cp, xpos, ypos = load_data(data_path)

    parameters = np.column_stack((Alpha, Mach))

    trainCount = int(np.floor(parameters.shape[0] * 0.8))

    if parameters.shape[0] > trainCount:
        np.random.seed(42)
        selected_indices = np.random.choice(parameters.shape[0], trainCount, replace=False)
        parameters = reduce_data(parameters, selected_indices, 0)
        Cp = reduce_data(Cp, selected_indices, 1)

    print("Samples:", parameters.shape[0])

    weights = compute_rbf_weights(parameters, Cp.T, epsilon, kernel)
    


    print("------ TEST 1 ------")
    interpolated_coefficients = rbf_interpolate(parameters, weights, T1, epsilon, kernel)
    reconstructed_cp = interpolated_coefficients
    calculations(reconstructed_cp.T, T1, T1OG, xpos, ypos)
    
    print("------ TEST 2 ------")
    interpolated_coefficients = rbf_interpolate(parameters, weights, T2, epsilon, kernel)
    reconstructed_cp = interpolated_coefficients
    calculations(reconstructed_cp.T, T2, T2OG, xpos, ypos)
    
    print("------ TEST 3 ------")
    interpolated_coefficients = rbf_interpolate(parameters, weights, T3, epsilon, kernel)
    reconstructed_cp = interpolated_coefficients
    calculations(reconstructed_cp.T, T3, T3OG, xpos, ypos)
    
    print("------ TEST 4 ------")
    interpolated_coefficients = rbf_interpolate(parameters, weights, T4, epsilon, kernel)
    reconstructed_cp = interpolated_coefficients
    calculations(reconstructed_cp.T, T4, T4OG, xpos, ypos)
    
    print("------ TEST 5 ------")
    interpolated_coefficients = rbf_interpolate(parameters, weights, T5, epsilon, kernel)
    reconstructed_cp = interpolated_coefficients
    calculations(reconstructed_cp.T, T5, T5OG, xpos, ypos)
    



if __name__ == '__main__':
    main()
    pp.show()
