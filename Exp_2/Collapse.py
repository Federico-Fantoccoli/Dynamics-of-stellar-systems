import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8,8]

# Number of particles
N = 15

#Initial time
t0 = 0 

# Masses of each particle
M = 1

# Radial distribution
x_r = np.random.uniform(0, 1, N)
# Radius
a = 1
r_points = a * x_r**(1/3)

# Phi distribution
phi_points = np.random.uniform(0, 2 * np.pi, N)

# Theta distribution
x_theta = np.random.uniform(0, 1, N)
theta_points = np.arccos(1 - 2*x_theta)

# Checking the distributions
plt.hist(r_points, bins = 30, density = True)
plt.plot(np.sort(x_r), 3* np.sort(x_r**2))
plt.show()
plt.hist(phi_points, bins = 30, density = True)
plt.show()
plt.hist(theta_points, bins = 30, density = True)
plt.show()

# checking the maximum
print(max(r_points), min(r_points))
print(max(theta_points), min(theta_points))
print(max(phi_points), min(phi_points), '\n')


def cartesian_coord (r, theta, phi):
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.array([x, y, z])

# Cartesian coordinates
cartesian = cartesian_coord(r_points, theta_points, phi_points)
cartesian = np.transpose(cartesian)
print(np.shape(cartesian))

print(max(cartesian[:,0]))
print(max(cartesian[:,1]))
print(max(cartesian[:,2]))

# Checking the sphere
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(cartesian[:,0], cartesian[:,1], cartesian[:,2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()

plt.scatter(cartesian[:,0], cartesian[:,1])
plt.title('XY')
plt.show()

plt.scatter(cartesian[:,0], cartesian[:,2])
plt.title('XZ')
plt.show()

plt.scatter(cartesian[:,1], cartesian[:,2])
plt.title('YZ')
plt.show()

# Every particle is generated at rest
velocity_coord_str = "0 0 0"

# Converts the list of cartesian coordinates into a list of strings
# and append the mass of the particle ad the beginning
# and the initial velocity at the end
points_coord_str = ['\n' + str(M) + ' ' +
                    ' '.join(str(coord) for coord in cartesian[i]) + ' ' +
                    velocity_coord_str
                    for i in range(N)]

# Create or overwrite the input file
file_name = "collapse_in.txt"

input_file = open(file_name, 'w')

# Write a file in the proper format for the nbody_sh1 program:
#
# N
# t_0
# m_1 x_1, y_1, z_1, vx_1, vy_1, vz_1
# m_2 x_2, y_2, z_2, vx_2, vy_2, vz_2
# ...
# m_N x_N, y_N, z_N, vx_N, vy_N, vz_N

input_file.write(str(N))
input_file.write('\n' + str(t0))
input_file.writelines(points_coord_str)

input_file.close()


