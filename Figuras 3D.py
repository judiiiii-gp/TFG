import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import griddata

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

def plot_cp_3d(xpos, ypos, zpos, Cp, Alpha, Mach, idx):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    print(Cp[:, idx])
    sc = ax.scatter(xpos, ypos, zpos, c=Cp[:, idx], cmap='viridis')
    fig.colorbar(sc, ax=ax, label='Cp')
    ax.set_title(f'Cp distribution (Alpha={Alpha[idx, 0]}, Mach={Mach[idx, 0]})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_cp_2d(xpos, Cp, Alpha, Mach, idx):

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(xpos, ypos, c=Cp[:, idx], cmap='viridis')
    plt.colorbar(scatter, label='$C_p$')
    plt.title(f'Distribución de $C_p$ en (x, y) - Alpha = {Alpha[idx, 0]}°, Mach = {Mach[idx, 0]}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    
def load_validation_file(directory_path, section):
    
    filename = f'cp_{section}.dat'
    filepath = os.path.join(directory_path, filename)
    
    # Leemos el archivo
    try:
        data = np.loadtxt(filepath)
        X = data[:, 0]
        Cp = data[:, 1]
        return X, Cp
    except FileNotFoundError:
        print(f"⚠️ Archivo '{filename}' no encontrado en '{directory_path}'.")
        return None, None
    except Exception as e:
        print(f"⚠️ Error al leer '{filename}': {e}")
        return None, None

def plot_cp_2d_xz(xpos, ypos, zpos, Cp, Alpha, Mach):
    labels = [99, 95, 90, 80, 65, 44, 20]
    
    #### FOM ######
    indexes    = zpos > 0
    x_fom      = xpos[indexes]
    y_fom      = ypos[indexes]
    cp_fom     = Cp[indexes]
    indexes    = zpos <= 0
    x_fom_inf  = xpos[indexes]
    y_fom_inf  = ypos[indexes]
    cp_fom_inf = Cp[indexes]
    
    #### ROM ######
    indexes    = zpos > 0
    cp_rom     = Cp[indexes]
    indexes    = zpos <= 0
    cp_rom_inf = Cp[indexes]

    #### RBF ######
    indexes    = zpos > 0
    cp_rbf     = Cp[indexes]
    indexes    = zpos <= 0
    cp_rbf_inf = Cp[indexes]
    
    for section in range(6):
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
        
        #VALIDATION#
        dir = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\BBDD 3D\\reference_data\\onera\\experiment"
        X, Cp = load_validation_file(dir, labels[section])
        plt.plot(X, Cp, '-', color = 'blue', linewidth = 1.7, label = "Exp.")
        
        # # #FOM#
        # cp_interpolated_fom = griddata((x_fom, y_fom), cp_fom, (x_grid, y_target), method='linear', fill_value=0.25)
        # cp_interpolated_fom_inf = griddata((x_fom_inf, y_fom_inf), cp_fom_inf, (x_grid, y_target), method='linear')
        # cp_interpolated_fom_full = np.concatenate((cp_interpolated_fom, cp_interpolated_fom_inf[::-1]))
        # plt.plot(x_airfoil_normalized_full, -cp_interpolated_fom_full, '-', color='blue', linewidth=1.5, label='FOM')

        # #### ROM ######
            
        # cp_interpolated_rom = griddata((x_fom, y_fom), cp_rom, (x_grid, y_target), method='linear')
        # cp_interpolated_rom_inf = griddata((x_fom_inf, y_fom_inf), cp_rom_inf, (x_grid, y_target), method='linear')
        # cp_interpolated_rom_full = np.concatenate((cp_interpolated_rom, cp_interpolated_rom_inf[::-1]))
        # plt.plot(x_airfoil_normalized_full, -cp_interpolated_rom_full, '-', color='red', linewidth=1.3, label=f"ROM")

        # #### RBF ######
        # cp_interpolated_rbf = griddata((x_fom, y_fom), cp_rbf, (x_grid, y_target), method='linear')
        # cp_interpolated_rbf_inf = griddata((x_fom_inf, y_fom_inf), cp_rbf_inf, (x_grid, y_target), method='linear')
        # cp_interpolated_rbf_full = np.concatenate((cp_interpolated_rbf, cp_interpolated_rbf_inf[::-1]))
        # plt.plot(x_airfoil_normalized_full, -cp_interpolated_rbf_full, '--', color='green', linewidth=1.1, label=f"RBF")

        plt.xlim(-0.1, 1.1)
        plt.ylim(-1.0, 1.3)

        # Set labels
        math_label_fontsize = 14
        x_label = r'$x$'
        y_label = r'$-C_p$'
        plt.xlabel(x_label, fontsize = math_label_fontsize)
        plt.ylabel(y_label, fontsize = math_label_fontsize)

        plt.legend(loc='upper right')
        plt.tight_layout()

data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\BBDD 3D\\FOM_Skin_Data"
Alpha, Mach, Cp, xpos, ypos, zpos = load_data(data_path)
parameters = np.column_stack((Alpha, Mach))
alfa = 0.15232
mach = 0.60535
parameters = np.round(parameters, 5)
idx = np.where((parameters[:, 0] == alfa) & (parameters[:, 1] == mach))[0]

plot_cp_3d(xpos, ypos, zpos, Cp, Alpha, Mach, idx)
plot_cp_2d(xpos, Cp, Alpha, Mach, idx)
plot_cp_2d_xz(xpos, ypos, zpos, Cp, Alpha, Mach)

plt.show()