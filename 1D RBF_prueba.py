import os
import numpy as np
import matplotlib.pyplot as plt

simulation = 1994
epsilon = 26.5
rbf_type = 1
points = 1000

#RBF function. If type = 0, the RBF function that will be applied will be linear, if not it will be gaussian.
def rbf(epsilon, distance, type):
    if type ==0:
        result = distance
    else:
        result = np.exp(-1*epsilon*epsilon*distance*distance)
    return result

#Function that performs the interpolation when we have new points
def rbf_interpolation(x, M_train, weights, epsilon, type):
    #Creation of the matrix where the new points will be stored
    Cp = np.zeros((len(x), 1))
    t = 0
    #Initialization of the loop, because we will need to interpolate for all the values of our new X
    while t<len(x):
        f = 0
        #We save the X value in position t
        x1 = x[t]
        #Initialization of the matrix where we will store the RBF results
        A = np.zeros((len(M_train), 1))
        while f<len(M_train):
            #We save the X value of the train matrix
            x2 = M_train[f, 0]
            #Calculation of the distance between the X train value and the X new point value
            distance = np.abs(x2-x1)
            #Performance of the RBF function
            A[f] = rbf(26.5, distance, 1)
            f+=1
        #To calculate the new Cp we need to transpose the Matrix A, with all the RBF results and multiply this matrix by the weight matrix
        Cp[t] = np.dot(A.T, weights)
        t+=1
    #The result is a matrix 1D (with one column)
    return Cp

#We make a list of all the archives in the directory
link = "C:\\Users\\judig\\OneDrive\\Escritorio\\TFG\\Exemple TFG\\CODE\\Naca0012_database_mesh_1\\FOM_Skin_Data"
DIR_LIST = os.listdir(link)

#We choose which simulation do we want to use
archive = DIR_LIST[simulation]

#We eliminate the file extension
archive_no_extension = archive.replace(".dat", "")

#We separate attack angle and Mach values
values = archive_no_extension.split(",")
attack_angle = values[0]
Mach = values[1]

print("The file we are reading has an attack angle = " + str(attack_angle) + "and Mach = " + str(Mach))
#We open the simulation file to read it
complete_link = os.path.join(link, archive)
file = open(complete_link, 'r')

#We read all the lines and save them in the lines variable
lines = file.readlines()

#We read line by line from the file and separate the top and bottom values
#The file's structure is: X Y Z Cp. All values are separated by spaces. 
#To see if the values correspond to the top or to the bottom we need to see if y is positive (top) or negative (bottom)
i=0
y_top = 0
y_bottom = 0
while i<len(lines):
    line = lines[i].split(' ')
    #We remove the new line character 
    line[3] = line[3].removesuffix('\n')
    y = float(line[1])
    if y<0:
        y_bottom +=1
    else:
        y_top +=1
    i+=1

#Creation of the matrices that will contain X and Cp values, that after the training wll be plotted to compare the result with the interpolated one.
X_Cp_top = np.zeros([y_top, 2])
X_Cp_bottom = np.zeros([y_bottom, 2])
#We add the values in the simulation file to the Matrix created
j=0
top=0
bottom=0
while j<len(lines):
    line = lines[j].split(' ')
    line[3] = line[3].removesuffix('\n')
    y = float(line[1])
    if y<0:
        X_Cp_bottom[bottom, 0] = line[0]
        X_Cp_bottom[bottom, 1] = line[3]
        bottom +=1
    else:
        X_Cp_top[top, 0] = line[0]
        X_Cp_top[top, 1] = line[3]
        top +=1
    j+=1

#We can create the linear system that we will need to solve to find the weights, as to apply RBF later on

#First we are going to calculate the weights of the top
N_top = len(X_Cp_top)
Cp_top = np.zeros((N_top, 1)) #This is the Y matrix of our linear system
A_top = np.zeros((N_top, N_top))  #This is the X matrix of our linear system
i = 0
#We will perform the RBF function for all possible combinations in our Matrices. Then we will compute the eights, that will tell us which are the most important values.
while i < N_top:
    j = 0
    while j< N_top:
        x_1 = X_Cp_top[i, 0]
        x_2 = X_Cp_top[j, 0]
        distance = np.abs(x_1-x_2)
        A_top[i, j] = rbf(26.5, distance, 1)
        j+=1
    Cp_top[i] = X_Cp_top[i, 1]
    i+=1
#We solve the linear system to find the weight matrix of the top 
weights_top = np.linalg.solve(A_top, Cp_top)

#Now we compute the bottom weights as we did previously with the top ones
N_bottom = len(X_Cp_bottom)
Cp_bottom = np.zeros((N_bottom, 1))
A_bottom = np.zeros((N_bottom, N_bottom)) 
i = 0
while i < N_bottom:
    j = 0
    while j< N_bottom:
        x_1 = X_Cp_bottom[i, 0]
        x_2 = X_Cp_bottom[j, 0]
        distance = np.abs(x_1-x_2)
        A_bottom[i, j] = rbf(26.5, distance, 1)
        j+=1
    Cp_bottom[i] = X_Cp_bottom[i, 1]
    i+=1

weights_bottom = np.linalg.solve(A_bottom, Cp_bottom)


#New points
#Now the computer is going to try to plot the top and bottom pressure profile of unknown points.

#First we create an X vector.
X_new = np.linspace(0,1, points)

#Now we want to find for each value in X, the value of Cp_top and Cp_bottom
#We eliminate the X values that are higher than 0.99 
X_new = X_new[X_new<0.99] 
#We perform the interpolation function to find the new Cp values from the top and bottom surfaces
Cp_top_new = rbf_interpolation(X_new, X_Cp_top, weights_top, epsilon, rbf_type)
Cp_bottom_new = rbf_interpolation(X_new, X_Cp_bottom, weights_bottom, epsilon, rbf_type)
print(Cp_top_new)

#We sort the values to plot them correctly
indices_bottom = np.argsort(X_Cp_bottom[:, 0])
X_Cp_bottom = X_Cp_bottom[indices_bottom, :]

indices_top = np.argsort(X_Cp_top[:, 0])
X_Cp_top = X_Cp_top[indices_top, :]

indices_bottom_new = np.argsort(X_new)
Cp_bottom_new = Cp_bottom_new[indices_bottom_new]

indices_top_new = np.argsort(X_new)
Cp_top_new = Cp_top_new[indices_top_new]


#We plot the values of the original simulation
plt.close()
plt.grid()
plt.plot(X_Cp_bottom[:, 0], X_Cp_bottom[:, 1], label ='Bottom surface', marker = 'x')
plt.plot(X_Cp_top[:, 0], X_Cp_top[:, 1], label ='Upper surface', marker = 'o')
plt.legend()
plt.xlabel('X')
plt.ylabel('Cp')
plt.title("Original simulation")
plt.gca().invert_yaxis()

plt.figure()
#We plot the values of the interpolated simulation

plt.grid()
plt.plot(X_new, Cp_bottom_new, label ='Bottom surface', marker = 'x')
plt.plot(X_new, Cp_top_new, label ='Upper surface', marker = 'o')
plt.legend()
plt.xlabel('X')
plt.ylabel('Cp')
plt.title("Interpolated simulation")
plt.gca().invert_yaxis()
plt.show()
plt.show()