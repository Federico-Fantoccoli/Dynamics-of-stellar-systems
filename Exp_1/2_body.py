import numpy as np
import matplotlib.pyplot as plt
orbits = np.genfromtxt("sun_earth.plot.txt", delimiter = ' ', invalid_raise=False)
t = 7
step = 0.01
m1 = 1.0 #solar mass IU
m2 = 0.000003 #earth mass IU

## internal units convertions ##
au = 149597870.707 #km
v_se_units = 30. #km/s
t_se_units = au / (v_se_units * 3.154e+7) #years

#print(orbits)
#n_step = int(t / step)

x_1 = np.zeros(shape = int(len(orbits)/2))
x_2 = np.zeros(shape = int(len(orbits)/2))
y_1 = np.zeros(shape = int(len(orbits)/2))
y_2 = np.zeros(shape = int(len(orbits)/2))

for i in range(int(len(orbits)/2)):
    x_1[i] = orbits[i*2][0]
    y_1[i] = orbits[i*2][1]
    x_2[i] = orbits[i*2 +1][0]
    y_2[i] = orbits[i*2 +1][1]

v_x_1 = np.zeros(shape = int(len(orbits)/2))
v_x_2 = np.zeros(shape = int(len(orbits)/2))
v_y_1 = np.zeros(shape = int(len(orbits)/2))
v_y_2 = np.zeros(shape = int(len(orbits)/2))

for i in range(int(len(orbits)/2)):
    v_x_1[i] = orbits[i*2][3]
    v_y_1[i] = orbits[i*2][4]
    v_x_2[i] = orbits[i*2 +1][3]
    v_y_2[i] = orbits[i*2 +1][4]

time = np.linspace(0, t*t_se_units, num=len(v_x_1))

maxi = []
for i in range (len(v_y_2)):
    if v_y_2[i] == v_y_2.max():
        maxi.append(i)
year = maxi[0] * step
actual_year = year * t_se_units
print("The year is approximately " + str((1 - actual_year)*365) + " days shorter than expected")

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
fig.suptitle('Sun-Earth Orbit')
ax1.scatter(x_1, y_1, color = 'orange', label = 'Sun', marker = '*')
ax1.scatter(x_2, y_2, color = 'green', s = 3, label = 'Earth')
ax1.set_xlabel('x [AU]')
ax1.set_ylabel('y [AU]')
ax1.set_title('Orbital motion in the $(x,y)$ plane')
ax1.legend()
ax1.axis('equal')

ax2.plot(time, v_y_2*v_se_units)
ax2.axvline(1., label='one expected full orbit', color='red')
ax2.axvline(actual_year, label='one actual full orbit', color='orange')
#ax2.plot(v_x_2)
ax2.set_xlabel('Time [years]')
ax2.set_ylabel(r'$v_y$ [km/s]')
ax2.set_title('Earth velocity along the $y$ axis')
ax2.legend()
#ax2.axis('equal')

peri = np.abs(x_2.min())
apo = np.abs(x_2.max())

e = (apo - peri)/(apo + peri)
actual_e = 0.0167
print('Found eccentricity:', e)
print('Actual eccentricity:', actual_e)

k1 = 0.5 * m1 * ((v_x_1)**2 + (v_y_1)**2)
k2 = 0.5 * m2 * ((v_x_2)**2 + (v_y_2)**2)

k_tot = k1 + k2

r = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
U = - (m1 * m2) / r
E_tot = k_tot + U

#plt.plot(time, k1)
#plt.plot(time, k2)
plt.plot(time, U, label='$U$')
plt.plot(time, k_tot, label='$K_{tot}$')
plt.plot(time, E_tot, label='$E_{tot}$')
plt.xlabel('Time [years]')
plt.ylabel('Energy [IU]')
plt.title('Energy of the system')
plt.legend()

r0 = np.zeros((len(x_1), 2)) #vettore del coo del centro di massa
r0[:, 0] = x_1 - x_2 #relative distance vector
r0[:, 1] = y_1 - y_2

v0 = np.zeros((len(x_1), 2))
v0[:, 0] = v_x_1 - v_x_2 #relative velocity vector
v0[:, 1] = v_y_1 - v_y_2

mu_red = (m1*m2)/(m1+m2)
L_z_com = mu_red*(r0[:, 0]*v0[:, 1] - r0[:, 1]*v0[:, 0])

def V_eff(x):
    return (np.mean(L_z_com)**2)/(2*mu_red*x**2) - m1*m2/x

x_dense = np.linspace(0.5*apo, 10*peri, 10000)

fig2, (ax_am, ax_effpot) = plt.subplots(1,2, figsize=(12,5))
ax_am.plot(time, L_z_com)
ax_am.set_ylim(2.989970e-6, 2.989972e-6) #praticamente costante, varia sensibilmente dalla dodicesima cifra decimale
ax_am.set_title('Total angular momentum of the system')
ax_am.axhline(np.mean(L_z_com), color='red')
ax_am.set_ylabel('$L$ [IU]')
ax_am.set_xlabel('Time [years]')

ax_effpot.plot(x_dense, V_eff(x_dense))
ax_effpot.set_title('Efficient potential of the system')
ax_effpot.set_ylabel('$L$ [IU]')
ax_effpot.set_xlabel('Relative position [IU]')
ax_effpot.set_ylim(-1.6e-6, 0.25e-6)

#######sun + halleys
orbits_h = np.genfromtxt("sun_halley.plot.txt", delimiter = ' ', invalid_raise=False)

t_h = 475
step_h = 0.1
m1_h = 1
m2_h = 1.1e-15
#print(orbits_2)

x_1h = np.zeros(shape = int(len(orbits_h)/2))
x_2h = np.zeros(shape = int(len(orbits_h)/2))
y_1h = np.zeros(shape = int(len(orbits_h)/2))
y_2h = np.zeros(shape = int(len(orbits_h)/2))

for i in range(int(len(orbits_h)/2)):
    x_1h[i] = orbits_h[i*2][0]
    y_1h[i] = orbits_h[i*2][1]
    x_2h[i] = orbits_h[i*2 +1][0]
    y_2h[i] = orbits_h[i*2 +1][1]

v_x_1h = np.zeros(shape = int(len(orbits_h)/2))
v_x_2h = np.zeros(shape = int(len(orbits_h)/2))
v_y_1h = np.zeros(shape = int(len(orbits_h)/2))
v_y_2h = np.zeros(shape = int(len(orbits_h)/2))

for i in range(int(len(orbits_h)/2)):
    v_x_1h[i] = orbits_h[i*2][3]
    v_y_1h[i] = orbits_h[i*2][4]
    v_x_2h[i] = orbits_h[i*2 +1][3]
    v_y_2h[i] = orbits_h[i*2 +1][4]

time_h = np.linspace(0, t_h*t_se_units, num=len(v_x_1h)) 

maxi_h = []
for i in range (len(v_y_2h)):
    if v_y_2h[i] == v_y_2h.max():
        maxi_h.append(i)

year_h = maxi_h[0] * step_h
actual_year_h = year_h * t_se_units

print("The full orbit is approximately " + str((76 - actual_year_h)*365) + " days shorter than expected")

fig1, (ax1_h, ax2_h) = plt.subplots(1,2, figsize=(14,5))
fig1.suptitle("Sun-Halley's comet Orbit")
ax1_h.scatter(x_1h, y_1h, color = 'orange', marker = '*', label = 'Sun' )
ax1_h.scatter(x_2h, y_2h, color = 'deepskyblue', s = 3, label = "Halley's Comet")
ax1_h.legend()
ax1_h.set_xlabel('x [AU]')
ax1_h.set_ylabel('y [AU]')
ax1_h.set_title('Orbital motion in the $(x,y)$ plane')
#plt.axis('equal')

ax2_h.plot(time_h, v_y_2h*v_se_units)
ax2_h.set_xlabel('Time [years]')
ax2_h.set_ylabel(r'$v_y$ [km/s]')
ax2_h.axvline(76., label='one expected full orbit', color='red')
ax2_h.axvline(actual_year_h, label='one actual full orbit', color='orange')
ax2_h.set_title("Halley's comet velocity along the $y$ axis")

peri_h = np.abs(x_2h.min())
apo_h = np.abs(x_2h.max())

e_h = (apo_h - peri_h)/(apo_h + peri_h)
actual_e_h = 0.96658
print('Found eccentricity:', e_h)
print('Actual eccentricity:', actual_e_h)

k1_h = 0.5 * m1_h * ((v_x_1h)**2 + (v_y_1h)**2)
k2_h = 0.5 * m2_h * ((v_x_2h)**2 + (v_y_2h)**2)

k_tot_h = k1_h + k2_h

r_h = np.sqrt((x_1h - x_2h)**2 + (y_1h - y_2h)**2)
U_h = - (m1_h * m2_h) / r_h
E_toth = k_tot_h + U_h

plt.plot(time_h, U_h, label='$U$')
plt.plot(time_h, k_tot_h, label='$K_{tot}$')
plt.plot(time_h, E_toth, label='$E_{tot}$')
plt.xlabel('Time [years]')
plt.ylabel('Energy [IU]')
plt.title('Energy of the system')
plt.legend()

r0_h = np.zeros((len(x_1h), 2)) #vettore del coo del centro di massa
r0_h[:, 0] = x_1h - x_2h #relative motion vector
r0_h[:, 1] = y_1h - y_2h

v0_h = np.zeros((len(x_1h), 2))
v0_h[:, 0] = v_x_1h - v_x_2h #relative velocity vector
v0_h[:, 1] = v_y_1h - v_y_2h

mu_red_h = (m1_h*m2_h)/(m1_h+m2_h)
L_z_com_h = mu_red_h*(r0_h[:, 0]*v0_h[:, 1] - r0_h[:, 1]*v0_h[:, 0])

def V_eff_h(x):
    return (np.mean(L_z_com_h)**2)/(2*mu_red*x**2) - m1*m2/x
x_dense_h = np.linspace(0.5*apo_h, 10*peri_h, 10000)

fig2_h, (ax_am_h, ax_effpot_h) = plt.subplots(1,2, figsize=(12,5))
ax_am_h.plot(time_h, L_z_com_h, label='$L(t)$')
#ax_am_h.set_ylim(1.156e-15, 1.367e-15) #praticamente costante, varia sensibilmente dalla dodicesima cifra decimale
ax_am_h.set_title('Total angular momentum of the system')
ax_am_h.axhline(np.mean(L_z_com_h), label='Average $L$', color='red')
ax_am_h.set_ylabel('$L$ [IU]')
ax_am_h.set_xlabel('Time [years]')
ax_am_h.legend()

ax_effpot_h.plot(x_dense_h, V_eff_h(x_dense_h))
ax_effpot_h.set_title('Efficient potential of the system')
ax_effpot_h.set_ylabel('$L$ [IU]')
ax_effpot_h.set_xlabel('Relative position [IU]')
#ax_effpot_h.set_ylim(-1.6e-6, 0.25e-6)

#####sun comet
orbits_c= np.genfromtxt("sun_comet.plot.txt", delimiter = ' ', invalid_raise=False)
x_1c = np.zeros(shape = int(len(orbits_c)/2))
x_2c = np.zeros(shape = int(len(orbits_c)/2))
y_1c = np.zeros(shape = int(len(orbits_c)/2))
y_2c = np.zeros(shape = int(len(orbits_c)/2))

for i in range(int(len(orbits_c)/2)):
    x_1c[i] = orbits_c[i*2][0]
    y_1c[i] = orbits_c[i*2][1]
    x_2c[i] = orbits_c[i*2 +1][0]
    y_2c[i] = orbits_c[i*2 +1][1]

plt.scatter(x_2c, y_2c, color = 'green', s = 3, label = 'Comet')
plt.scatter(x_1c, y_1c, color = 'orange', label = 'Sun', marker = '*')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Hyperbolic motion in the $(x,y)$ plane')
plt.legend()
plt.axis('equal')

