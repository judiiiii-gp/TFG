import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import matplotlib.tri as mtri
import matplotlib.cm as cm
from scipy.spatial import Delaunay, ConvexHull
from scipy.integrate import simpson

#Function to load the simulation data
def load_data(data_path):
    
    #We list all the files in the directory
    data_list = os.listdir(data_path)
    num_files = len(data_list)
    
    
    sample_file = os.path.join(data_path, data_list[0])
    with open(sample_file, 'r') as f:
        num_points = len(f.readlines())

    #Creation of the variables where the data will be stored
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
                #X, Y and Z are the same for all the simulations, so we only save them in the first iteration
                if i == 0:
                    xpos[j] = x
                    ypos[j] = y
                    zpos[j] = z

    return Alpha, Mach, Cp, xpos, ypos, zpos

#Function to plot the pressure distribution in 3D around the airfoil
def plot_cp_3d(xpos, ypos, zpos, Cp, Alpha, Mach):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(xpos, ypos, zpos, c=Cp, cmap='viridis')
    fig.colorbar(sc, ax=ax, label='Cp')
    ax.set_title(f'Cp distribution (Alpha={Alpha[idx, 0]}, Mach={Mach[idx, 0]})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.view_init(elev=30, azim=135) 

    ax.set_xlim(xpos.min(), xpos.max())
    ax.set_ylim(ypos.min(), ypos.max())
    ax.set_zlim([-0.25, 0.25])
    plt.tight_layout()

#Function to plot the pressure distribution around the airfoil in 2D (plane xy)
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
    plt.title(f'Cp distribution (Alpha={Alpha[idx, 0]}, Mach={Mach[idx, 0]})')
    plt.tight_layout()

#Function to load the file to validate the results
def load_validation_file(directory_path, section):
    
    filename = f'cp_{section}.dat'
    filepath = os.path.join(directory_path, filename)
    
    # We reed the file
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

#Function to plot the pressure distribution from different sections of the airfoil. The plot is in 2D.
def plot_cp_2d_xz(xpos, ypos, zpos, Cp, Alpha, Mach):
    labels = [95, 90, 80, 65, 44, 20]
    

    #### FOM ######
    x_fom_up = []
    y_fom_up = []
    z_fom_up = []
    cp_fom_up = []
    x_fom_down = []
    y_fom_down = []
    z_fom_down = []
    cp_fom_down = []
    
    i = 0
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
        X, Cp_val = load_validation_file(dir, labels[section])
        plt.plot(X, Cp_val, '-', color = 'green', linewidth = 1.7, label = "Exp.")
        
        # # #FOM#
        cp_interpolated_fom = griddata((x_fom_up, y_fom_up), cp_fom_up, (x_grid,y_target), method='linear', fill_value=0.25)
        cp_interpolated_fom_inf = griddata((x_fom_down, y_fom_down), cp_fom_down, (x_grid,y_target), method='linear', fill_value=0.25)
        cp_interpolated_fom_full = np.concatenate((cp_interpolated_fom, cp_interpolated_fom_inf[::-1]))
        plt.plot(x_airfoil_normalized_full, -cp_interpolated_fom_full, '-', color='blue', linewidth=1.5, label='FOM')

        
        plt.xlim(-0.1, 1.1)
        plt.ylim(-1.0, 1.3)

        # Set labels
        math_label_fontsize = 14
        x_label = r'$x$'
        y_label = r'$-C_p$'
        plt.xlabel(x_label, fontsize = math_label_fontsize)
        plt.ylabel(y_label, fontsize = math_label_fontsize)
        plt.title("Plot section y = " + str(labels[section]))
        plt.legend(loc='upper right')
        plt.tight_layout()

#Function to plot the comparison between the simulated Cp and the interpolated Cp
def plot_cp(Cp_real, Cp_interpolado, text):
    plt.figure()
    plt.scatter(Cp_real, Cp_interpolado, color='blue', label='Cp interpolado vs Cp real')
    plt.plot([min(Cp_real), max(Cp_real)], [min(Cp_real), max(Cp_real)], color='red', linestyle='--', label='Interpolación perfecta (y = x)')

    plt.xlabel('Cp real')
    plt.ylabel('Cp interpolado')
    plt.title('Comparación de Cp real vs Cp interpolado ' + text)
    plt.legend()
    plt.grid(True)

#Function to separate the Cp values from the ones on the upper surface of the wing and the ones on the lower surface
def separate_Cp(xpos, ypos, zpos, Cp):
    i = 0
    
    #Creation of the variables
    x_fom_up = []
    y_fom_up = []
    z_fom_up = []
    cp_fom_up = []
    x_fom_down = []
    y_fom_down = []
    z_fom_down = []
    cp_fom_down = []
    
    #Loop through all the values to assign them on the corresponding surface
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

#Function that integrates the Cp along the airfoil surface
def integrate_surface(x, y, z, cp, is_upper):

    pts2d = np.vstack((x, y)).T
    tri = Delaunay(pts2d)

    lift = 0.0
    for tri_idx in tri.simplices:
        # points in 3D
        p1 = np.array([x[tri_idx[0]], y[tri_idx[0]], z[tri_idx[0]]])
        p2 = np.array([x[tri_idx[1]], y[tri_idx[1]], z[tri_idx[1]]])
        p3 = np.array([x[tri_idx[2]], y[tri_idx[2]], z[tri_idx[2]]])

        # Area-vector (area's module in z)
        dA_vec = np.cross(p2 - p1, p3 - p1) / 2.0
        area = abs(dA_vec[2])

        cp_avg = cp[tri_idx].mean()

        # the sign depends on the surface:
        # in the upper surface, dA_z > 0, its normal is pointed towards +z and the force is -Cp * area
        # in the upper surface, dA_z < 0, its normal is pointed towards -z and the force is +Cp * area 
        if is_upper:
            lift += -cp_avg * area
        else:
            lift +=  cp_avg * area

    return lift

#Function to compute the lift coefficient
def compute_cl(x, y, z, cp, S_ref):

    # Separation of the upper surface values from the lower surface values
    x_up,  y_up,  z_up,  cp_up, \
    x_low, y_low, z_low, cp_low = separate_Cp(x, y, z, cp)

    # Integration in each surface
    L_up   = integrate_surface(x_up,  y_up,  z_up,  cp_up, True)
    L_low  = integrate_surface(x_low, y_low, z_low, cp_low, False)

    # Lift and lift coefficient
    L_net  = L_up + L_low
    CL     = L_net / S_ref
    return CL

############## SCRIPT ##############################

#Load the data from the files
data_path = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\BBDD 3D\\FOM_Skin_Data"
Alpha, Mach, Cp, xpos, ypos, zpos = load_data(data_path)
parameters = np.column_stack((Alpha, Mach))

#Input values that are going to be studied
alfa = 3.06
mach = 0.839
parameters = np.round(parameters, 5)
idx = np.where((parameters[:, 0] == alfa) & (parameters[:, 1] == mach))[0]
cp_case = Cp[:, idx] 

#Distribution plots for the chosen case
plot_cp_3d(xpos, ypos, zpos, cp_case, Alpha, Mach)
plot_cp_2d(xpos, ypos, cp_case, Alpha, Mach)
plot_cp_2d_xz(xpos, ypos, zpos, cp_case, Alpha, Mach)

#Definition of the surface
S=0.7532

cp_case = cp_case.flatten()
Cl = compute_cl(xpos, ypos, zpos,  cp_case, S)
print(Cl)


plt.show()