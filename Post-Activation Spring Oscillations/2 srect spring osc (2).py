# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 18:18:39 2023

@author: Juke
"""


import numpy as np
from numpy import save
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
from scipy import interpolate
from PIL import Image
from mpl_toolkits import mplot3d
#plt.style.use(['default'])
plt.style.use(['dark_background'])

#%%

path = 'C:\\Users\\Juke\\Desktop\\Zelda Data\\Spring\\Oscillate\\2_srect_2\\'
file = open(path + 't_y.txt', 'r')
file2 = open(path + 't_y2.txt', 'r')
#read = file.read()

lines = file.readlines()
lines2 = file2.readlines()

#%%

t = lines[0:][::7]
y = lines[3:][::7]
y2 = lines2[3:][::7]

t=np.array(t)
theta = np.array(y)

T = []
for tt in t:
    T.append(int(tt.split()[0][6:][:-7]))

Y = []

for yy in y:
    Y.append(float(yy.split()[0][6:][:-7]))

Y2 = []

for yy in y2:
    Y2.append(float(yy.split()[0][6:][:-7]))


y = np.array(Y)
y2 = np.array(Y2)

dy = np.mean(y2 - y[:y2.size])
refH = 0.8198

#(dy)(scale)=reference height
scale = refH / dy

y = y * scale
y = y - y[-1]



t = np.array(T)#[:theta.size]
t = (t - t[0])/60

data=np.array([t, y])

plt.scatter(t, y)
plt.hlines(0, 0, t[-1])

dt = 1/60

ts = np.arange(0, t[-1] * 60 + 1, 1) * dt
f = interpolate.interp1d(t, y, kind='cubic')
ys = f(ts)

plt.plot(ts, ys)

tstart = 2.358

yf = y[t>tstart]
tf = t[t>tstart]
tf = tf-tf[0]

#%%

g = 7.25

M_rect = 3200

M_piston = 250

M = 2 * M_rect + M_piston


#%%

def DESol2(T, k, L, y0, v0, N):

    
    
    F = lambda y, v: -k * y - L * v
    
    a0 = F(y0, v0) / M
    
    sol = [[y0, v0, a0]]
    
    dt = 1/(60 * N)
    
    DT = np.diff(T)
    
    for i in range(DT.size):
        
        n = int(np.round(DT[i] * 60)) * N
        
        X0, V0, A0 = sol[i]
        
        temp = [[X0, V0, A0]]
        
        for j in range(1, n+1):
            
            x0, v0, a0 = temp[j-1]
            
            x1 = x0 + v0 * dt
            v1 = v0 + a0 * dt
            a1 = F(x1, v1) / M
        
            temp.append([x1, v1, a1])
            
        sol.append(temp[-1])
        
    sol = np.array(sol)
    
    return sol.T
#%% Perform curve fit and plot

ts = np.arange(0,tf[-1], 1/60)

f = lambda t, k, L, y0, v0, ye: DESol2(t, k, L, y0, v0, 1)[0] + ye

v1 = (yf[1]-yf[0]) / (tf[1]-tf[0])

popt, pcov= curve_fit(f, tf, yf, p0 = [100000, 5600, yf[0], v1, 0])

k, L, y0, v0, ye = popt

dk = pcov[0][0]**0.5
dL = pcov[1][1]**0.5

theta, omega, alpha = DESol2(ts, k, L, y0, v0, 1)+ye


fig, ax = plt.subplots(figsize = (16,8))
#ax.set_title('', fontsize = 20)


tp=[]
pk=[]

# find peaks
for i in range(1, theta.size-2):
    if (theta[i] - theta[i-1])*(theta[i+1] - theta[i]) < 0:
        tp.append(ts[i])
        pk.append(theta[i])

tp = np.array(tp)
pk = np.array(pk)

tpp = tp[pk>0]
pkp = pk[pk>0]        

tpn = tp[pk<0]
pkn = pk[pk<0]    

# fit peaks to exponentials
ex = lambda t, a, b: a*np.exp(-b*t)
A1, B1 = curve_fit(ex, tpp, pkp)[0]
A2, B2 = curve_fit(ex, tpn, pkn)[0]
       
ax.scatter(tpp,pkp, color = 'y')
ax.scatter(tpn,pkn, color = 'r')

ax.plot(ts, ex(ts, A1, B1))
ax.plot(ts, ex(ts, A2, B2))

ax.scatter(tf, yf, s=10, color = 'g', label = 'Measured data')
ax.plot(ts, theta, color = 'b', label = 'Model')
#ax.text(0.7, 0.5, '$\\theta$ = $35^o$', fontsize = 20, transform = ax.transAxes)

ax.set_xlabel('t [s]', fontsize = 20)
ax.set_ylabel('$z$ [m]', fontsize = 20)
ax.set_title('Spring Oscillations', fontsize = 20)

modeltx = 'Force Model: $F = -kz-\\lambda \\frac{dz}{dt}$'
ktx = '\n k= %.f  $\\pm$ % .f Zd/$s^2$' % (k, dk)
Ltx = '\n $\\lambda$=%.f  $\\pm$ % .1r Zd/s' % (L, dL)
ax.text(0.5, 0.8,  modeltx + ktx + Ltx,
        transform = ax.transAxes, fontsize = 20)

ax.set_xlim(0,tf[-1])


ax.hlines(0, 0, 30, linewidth = 1, alpha = 0.5)

ax.legend(loc = 'lower right', fontsize = 20)


#%% Energy

KE = 0.5 * M * omega ** 2
PE = 0.5 * k * theta ** 2
Etotal = KE + PE

plt.plot(ts, KE, color = 'b')
plt.plot(ts, PE, color = 'g')
plt.plot(ts, Etotal, color = 'y')
#%% Plot Fourier transform

fig3, ax = plt.subplots()
dt = 1/60
# fft needs evenly spaced points, so we interpolate the measured data
f = interpolate.interp1d(tf, yf - ye, kind='cubic')
N = ts.size
yF = fft(f(ts))
xf = fftfreq(N, dt)[:int(N/2)]
ax.plot(xf, 2.0/N * np.abs(yF[0:N//2]), label = '$\\mathcal{F}$(data)')



ax.set_xlabel('Frequency [hz]')
ax.set_ylabel('Frequency Amplitude $\\left[\\frac{Degrees}{hz}\\right]$')
ax.set_title('Fourier Transform')
ax.legend()

#%% Plot phase diagram 

v = np.diff(y, append=0)/np.diff(t, append=0)

#a, b = curve_fit(lambda t, a, b: a - b * t, t, v)[0]

#line = a - b * t

fig2, ax = plt.subplots(figsize = (10,12))
ax.scatter(y, v, color = 'g', label = 'Measured Data')
ax.plot(theta, omega, color = [0, 0, 1], label = 'Model')
#ax.plot(t, a - b * t, color = 'r')
ax.set_xlabel('$y$ [cm]', fontsize = 20)
ax.set_ylabel("$v$ [cm/s]", fontsize = 20)
ax.set_title('Phase Diagram', fontsize = 20)
ax.legend()
plt.tight_layout()

#%% 3d phase diagram

fig2 = plt.subplots()
ax = plt.axes(projection='3d')
ax.plot3D(theta, omega, alpha, 'green')
plt.show()

#%% animate full motion


dt = 1/60


fig, ax = plt.subplots(figsize=(20,10))


trace, = ax.plot([], [], '.', color = 'g')

ax.set_xlim( -0.1, t[-1])
ax.set_ylim( -1.9, 1.9)
#ax.set_xlabel('t [s]', fontsize = 20)
#ax.set_ylabel('$z$ [m]', fontsize = 20)
#ax.hlines(800, -0.2, t[-1])


fig.subplots_adjust(left = 0.1, right=0.9)


ax.tick_params(color='k', labelcolor='k')
for spine in ax.spines.values():
        spine.set_edgecolor('k')

delay = 60



def animate(i):
   
    trace.set_data(t[t<=i*dt], y[t<=i*dt])
 
ani = animation.FuncAnimation(fig, animate, interval=30, frames=ys.size, repeat = False)


ani.save(filename = path + 'FullMotion_b.mp4', writer='ffmpeg', fps=60)

#%% Zoom to fit data


tstart = 2.358
dt = 1/60


fig, ax = plt.subplots(figsize=(20,10))


ax.plot(t[t>tstart], y[t>tstart], '.', color = 'g')


yl_old = 1.9
xl_old = -0.1
ax.set_xlim( xl_old , t[-1])
ax.set_ylim( -yl_old, yl_old)
#ax.set_xlabel('t [s]', fontsize = 20)
#ax.set_ylabel('$z$ [m]', fontsize = 20)
#ax.hlines(800, -0.2, t[-1])


fig.subplots_adjust(left = 0.1, right=0.9)



ax.tick_params(color='k', labelcolor='k', labelsize = 20)
for spine in ax.spines.values():
        spine.set_edgecolor('k')

delay = 30

yl_new = 0.5
xl_new = tstart
N = 30

def animate(i):
    
    p1 = i / N
    p0 = 1-p1
    yl_i = p0 * yl_old + p1 * yl_new
    
    ax.set_ylim(-yl_i, yl_i )
    
    xl_i = p0 * xl_old + p1 * xl_new
    ax.set_xlim(xl_i, t[-1] )
    
    Color = [i/N, i/N, i/N]
    
    #ax.tick_params(color=Color, labelcolor=Color)
    for spine in ax.spines.values():
            spine.set_edgecolor(Color)
 
ani = animation.FuncAnimation(fig, animate, interval=17, frames=N+1, repeat = False)


#ani.save(filename = path + 'zoom.mp4', writer='ffmpeg', fps=60)

#%% animate curve fit


tstart = 2.358
yl_new = 0.5
dt = 1/60


fig, ax = plt.subplots(figsize=(20,10))


ax.plot(tf, yf, '.', color = 'g', label = 'Measured Data')
trace, = ax.plot([], [], lw = 2, color = [0,0.2,1], label = 'Model')

amp1, = ax.plot([], [], '--',  lw = 2, color = 'y')
amp2, = ax.plot([], [], '--',  lw = 2, color = 'y')

ax.set_xlim( 0 , tf[-1])
ax.set_ylim( -yl_new, yl_new)
#ax.set_xlabel('t [s]', fontsize = 20)
#ax.set_ylabel('$z$ [m]', fontsize = 20)
#ax.hlines(800, -0.2, t[-1])
ax.set_xlabel('t [s]', fontsize = 20)
ax.set_ylabel('$z$ [m]', fontsize = 20)
ax.set_title('Spring Oscillations', fontsize = 20)

modeltx = 'Force Model: $F$ = $m\\frac{d^2z}{dt^2}$ = $-kz-\\lambda \\frac{dz}{dt}$'
mtx = '\n m = %.f Zd' % (M)
ktx = '\n k = {:,}  $\\pm$ {} Zd/$s^2$'.format(int(round(k, -1)), int(round(dk, -1)))
Ltx = '\n $\\lambda$ = %.f  $\\pm$ % .1r Zd/s' % (L, dL)
txt = ax.text(0.5, 0.7,  modeltx + mtx + ktx + Ltx,
        transform = ax.transAxes, fontsize = 20, alpha = 0)
words = "This is essentially an exponentially decaying cosine, \n only I've integrated the forces once per frame \n rather than using the exact solution"
txt2 = ax.text(0.35, 0.2,  words,
        transform = ax.transAxes, fontsize = 20, alpha = 0)
ax.hlines(0, 0, 30, linewidth = 1, alpha = 0.5)

ax.legend(loc = 'lower right', fontsize = 20)

fig.subplots_adjust(left = 0.1, right=0.9)

'''
ax.tick_params(color='k', labelcolor='k')
for spine in ax.spines.values():
        spine.set_edgecolor('k')
'''
ax.tick_params(labelsize = 20)

N = ts.size

def animate(i):
    
    trace.set_data(ts[ts<=i*dt], theta[ts<=i*dt])
    txt.set_alpha(min(1, 2*i/N))
    if i > N/2:
        txt2.set_alpha(min(1, 2*i/N-1))
        
        tt = ts[ts<(i-N/2)*dt]
        amp1.set_data(tt, ex(tt, A1, B1))
        amp2.set_data(tt, ex(tt, A2, B2))
 
ani = animation.FuncAnimation(fig, animate, interval=17, frames=int(N*1.5), repeat = False)


#ani.save(filename = path + 'fit4.mp4', writer='ffmpeg', fps=60)

#%% animate z, z', z''


fig, ax = plt.subplots(figsize=(20,10))



tracez, = ax.plot([], [], lw = 2, color = 'g', label = 'Position')
tracezp, = ax.plot([], [], lw = 2, color = 'b', label = 'Velocity')
tracezpp, = ax.plot([], [], lw = 2, color = 'r', label = 'Acceleration')




ax.set_xlim(0, ts[-1])
ax.set_ylim(np.min(alpha), np.max(alpha))

ax.legend(loc = 'upper right', fontsize = 20)

fig.subplots_adjust(left = 0.1, right=0.9)


'''
ax.tick_params(color='k', labelcolor='k')
for spine in ax.spines.values():
        spine.set_edgecolor('k')
'''
ax.tick_params(labelsize = 20, labelcolor = 'k')

N = ts.size

def animate(i):
    
    tracez.set_data(ts[ts<=i*dt], theta[ts<=i*dt])
    tracezp.set_data(ts[ts<=i*dt], omega[ts<=i*dt])
    tracezpp.set_data(ts[ts<=i*dt], alpha[ts<=i*dt])
    
ani = animation.FuncAnimation(fig, animate, interval=17, frames=N, repeat = False)


#ani.save(filename = path + 'Energy2.mp4', writer='ffmpeg', fps=60)

#%% animate energy





fig, ax = plt.subplots(figsize=(20,10))



traceKE, = ax.plot([], [], lw = 2, color = [0,0.2,1], label = 'Kinetic Energy = $\\frac{1}{2}mv^2$')
tracePE, = ax.plot([], [], lw = 2, color = 'g', label = 'Potential Energy = $\\frac{1}{2}kz^2$')
traceT, = ax.plot([], [], lw = 2, color = 'y', label = 'Total Energy')

ax.set_xlabel('t [s]', fontsize = 20)
ax.set_ylabel('$Energy$ [Zd $\\frac{m^2}{s^2}$]', fontsize = 20)
ax.set_title('Energy Analysis', fontsize = 20)

ax.set_xlim(0, ts[-1])
ax.set_ylim(0, np.max(Etotal))

words = "The total energy should never increase like this, but the rough numerical integration \n introduces errors that violate energy conservation"
txt = ax.text(0.2, 0.5, words, transform = ax.transAxes, fontsize = 20, alpha = 0)
ax.legend(loc = 'upper right', fontsize = 20)

fig.subplots_adjust(left = 0.1, right=0.9)

ar = ax.arrow(0.2, 0.5, -0.06*1.2, -0.1*1.2, transform = ax.transAxes, alpha = 0)

'''
ax.tick_params(color='k', labelcolor='k')
for spine in ax.spines.values():
        spine.set_edgecolor('k')
'''
ax.tick_params(labelsize = 20)

N = ts.size

def animate(i):
    
    traceKE.set_data(ts[ts<=i*dt], KE[ts<=i*dt])
    tracePE.set_data(ts[ts<=i*dt], PE[ts<=i*dt])
    traceT.set_data(ts[ts<=i*dt], Etotal[ts<=i*dt])
    
ani = animation.FuncAnimation(fig, animate, interval=17, frames=N, repeat = False)


#ani.save(filename = path + 'Energy2.mp4', writer='ffmpeg', fps=60)

#%% animate phase diagram

x = []
y = []
rg = 1.2

thmax = np.max(theta)
wmax =np.max(omega)

fig, ax = plt.subplots(figsize=(5,5))

fig.subplots_adjust(left = 0.2, right=0.8, top = 0.8, bottom = 0.2)

trace, = ax.plot([], [], lw = 2, color = 'g')

ax.set_xlim( -rg*thmax*0.8, rg*thmax)
ax.set_ylim(-rg*wmax*1.2, rg*wmax)
ax.set_xlabel('$z$ [m]', fontsize = 20)
ax.set_ylabel("Velocity [m/s]", fontsize = 20)
ax.set_title('Phase Diagram', fontsize = 20)


S = 1
def animate(i):
    
    trace.set_data(theta[:S*i], omega[:S*i])
    
ani = animation.FuncAnimation(fig, animate, interval=17, frames=int(theta.size/S), repeat = False)

#ani.save(filename = path + 'Phase.mp4', writer='ffmpeg', fps=60)