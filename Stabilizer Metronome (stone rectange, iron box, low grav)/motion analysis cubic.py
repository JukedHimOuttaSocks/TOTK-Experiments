"""
Analysis of Metronome motion

Created on Sat Oct 28 22:28:12 2023

@author: Juke
"""

import numpy as np
from numpy import savetxt, load, save

import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy import interpolate
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

plt.style.use(['dark_background'])

#%% Edit this to the folder containing the t_theta.npy file

path = 'C:\\Users\\juke\\Desktop\\Zelda\\stabilizer torque\\dynamic\\srect_ibox_diag_LG\\'

#%% Load the measured data

data = load(path + 't_theta.npy')
t, th = data

#%% Constants (found in the datamined spreadsheets)

g = 7.25

M_box = 40000
I_box = 2.666667
R_box = 8 + 4 / 2 ** 0.5 # COM distance from axis
I_1 = M_box * (I_box + R_box ** 2) 

M_rect = 3200
I_rect = 5.491742
R_rect = 4
I_2 = M_rect * (I_rect + R_rect ** 2)

I = I_1 + I_2 # Total rotational inertia

#%% Differential equation solver

# Conversion factor, degrees to radians
Rd = np.pi/180

def DESol(t, A, B, C, th0, w0):

    S0 = [th0, w0] # initial conditions
    
    def dSdt(S, t):

        theta, omega = S 
        
        # return [d theta/dt, d omega/dt]
        return [omega,
                
        (
            g * (M_box * R_box + M_rect * R_rect)* np.sin(theta*Rd) # gravity
         
         - A * theta**3 # stabilizer torque
         
         - C * theta
         
         - B * omega # total angular damping 
         
         ) / I] # / total rotational inertia

    sol = odeint(dSdt, y0=S0, t=t)

    return sol.T

#%% Find best fit parameters

p = [ 5000, 800000, 10^6, th[0], (th[1]-th[0]) / (t[1] - t[0])] # initial guess

f = lambda t, A, B, C, th0, w0: DESol(t, A, B, C, th0, w0)[0]

A, B, C, th0, w0 = curve_fit(f, t, th, p0=p)[0]

#%% Plot data and curve fit

dt = 0.01
ts = np.arange(0, t[-1], dt)
theta, omega = DESol(ts, A, B, C, th0, w0)

fig, ax = plt.subplots(figsize = (12,6))

ax.set_title('Stabilizer Torque Model: $\\tau$ = $-k \\theta^3 -\\beta\\theta - \\alpha \\frac{d\\theta}{dt}$', fontsize = 20)

ax.set_xlabel('t [s]', fontsize = 20)
ax.set_ylabel('$\\theta$ [degrees]', fontsize = 20)

ax.set_xlim(0,30)
ax.set_ylim(-60,60)

ax.scatter(t, th, s=5, color = 'g', label = 'Measured Data')
ax.plot(ts, theta, color = 'b', label = 'Curve Fit')

ax.legend(fontsize = 20)

ax.hlines(0, 0, 30)

ax.text(0.7, 0.2, 
        ' k = %.f $\\tau$/deg$^3$ \n $\\beta$ = % .f $\\tau$/deg \n$\\alpha$ = %.f $\\frac{\\tau}{deg/s}$ ' % (A, C, B),
        fontsize = 20, transform = ax.transAxes)

#%% Plot phase diagram (angular position, angular velocity)

fig2, ax = plt.subplots(figsize = (10,12))
ax.scatter(th, np.diff(th, append=0)/np.diff(t, append=0), color = 'g', s=5, label = 'Measured Data')
ax.plot(theta, omega, color = [0, 0, 1], label = 'Model')
ax.set_xlabel('$\\theta$ [Degrees]', fontsize = 20)
ax.set_ylabel("$\\omega$ [Degrees/s]", fontsize = 20)
ax.set_title('Phase Diagram', fontsize = 20)
ax.legend()
plt.tight_layout()

#%% Torque Component Comparison

fig, ax = plt.subplots(figsize = (12,6))

grav = (g * (M_box * R_box + M_rect * R_rect)* np.sin(theta*Rd) )
torq3 =  -A * theta**3
torq1 = -C * theta
damp = - B * omega
total = grav + torq3 + torq1 + damp

ax.set_xlabel('t [s]', fontsize = 20)
ax.set_ylabel('$\\tau$ [$Zd$ $m^2$/$s^2$ ]x$10^8$', fontsize = 20)

ax.set_xlim(0,30)
ax.set_title('Torque Component Comparison')

ax.plot(ts, grav, color = 'g', label = 'Gravity')
ax.plot(ts, torq3 , color = 'b', label = 'Cubic Torque')
ax.plot(ts, torq1 , color = 'y', label = 'Linear Torque')
ax.plot(ts, damp, color = 'r', label = 'Damping')
ax.plot(ts, total, color = 'tab:purple', label = 'Total' )

ax.legend(loc = 'lower right', fontsize = 15)

#%% These don't mean very much but look cool anyway

fig2, axes = plt.subplots(1,2)

ax = axes[0]
ax.set_xlabel('$\\theta$', fontsize = 20)
ax.set_ylabel('Stabilizer Torque + Damping', fontsize = 20)

ax.plot((theta), (total), color = 'g')

ax = axes[1]
ax.set_xlabel('$\\omega$', fontsize = 20)
ax.set_ylabel('Stabilizer Torque Only', fontsize = 20)
ax.plot((omega), abs(torq3+torq1), color = 'g')

#%% Plot Fourier transform

fig3, ax = plt.subplots(figsize = (12,8))

# fft needs evenly spaced points, so we interpolate the measured data
f = interpolate.interp1d(t, th, kind='cubic')
N = ts.size
yf = fft(f(ts))
xf = fftfreq(N, dt)[:int(N/2)]
ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]), label = '$\\mathcal{F}$(data)', color = 'g')

f_max_i = np.argmax(np.abs(yf[0:N//2]))
f_max = xf[f_max_i]
ax.text(0.7, 0.7, '$f_{max}$ = %.3f Hz' % f_max, transform = ax.transAxes,
        fontsize = 20)

ax.set_xlabel('Frequency [hz]', fontsize = 20)
ax.set_xlim(0,1)
ax.set_ylabel('Frequency Amplitude $\\left[\\frac{Degrees}{hz}\\right]$', fontsize = 20)
ax.set_title('Fourier Transform', fontsize = 20)
ax.legend(fontsize = 20)

#%% Animate theta vs t

dt = 0.03333333
ts = np.arange(0, t[-1], dt)
theta, omega = DESol(ts, A, B, C, th0, w0)

x = []
y = []



fig, ax = plt.subplots(figsize=(20,10))

ax.set_title('Stabilizer Metronome: Tilt angle vs t', fontsize = 20,
             alpha = 1)

ax.set_xlabel('t [s]', fontsize = 20)
ax.set_ylabel('$\\theta$ [Degrees]', fontsize = 20)

ax.set_xlim( 0, ts[-1])
ax.set_ylim(-60,60)

trace, = ax.plot([], [], lw = 2, color = 'g', label = 'Measured data')

ax.hlines(0, 0, 30)

ax.legend(fontsize = 20)

def animate(i):
    
    
    x.append(t[i])
    y.append(th[i])
    
    trace.set_data(x, y)
    
ani = animation.FuncAnimation(fig, animate, interval=30, frames=theta.size, repeat = False)

#ani.save(path + 'quad.gif', writer='pillow', fps=30, dpi=100)

#%% Animate Curve fit

dt = 0.03333333
ts = np.arange(0, t[-1], dt)
theta, omega = DESol(ts, A, B, th0, w0)

x = []
y = []



fig, ax = plt.subplots(figsize=(20,10))

ax.plot(t, th, color = 'g', label = 'Measured data')

trace, = ax.plot([], [], lw = 2, color = 'b', label = 'Model')

ax.set_xlim( 0, ts[-1])
ax.set_ylim(-60,60)
ax.set_xlabel('t [s]', fontsize = 20)
ax.set_ylabel('$\\theta$ [Degrees]', fontsize = 20)
tit = ax.set_title('Stabilizer Torque Model: $\\tau$ = $-k \\theta^2 sign(\\theta) - \\alpha \\frac{d\\theta}{dt}$', fontsize = 20,
             alpha = 0)
txt = ax.text(22, -40,
        ' k = %.f thousand $\\frac{\\tau}{deg^2}$ \n\n $\\alpha$ = %.f thousand $\\frac{\\tau}{deg/s}$ ' % (A/1000, B/1000),
        fontsize = 20, alpha = 0)
ax.hlines(0, 0, 30)

ax.plot(ts, 35*np.ones(ts.size), '--', color = 'r')
ax.plot(ts, -35*np.ones(ts.size), '--', color = 'r')

ax.legend(fontsize = 20)
fig.subplots_adjust(bottom = 0.14)
def animate(i):
    
    tit.set_alpha(min(1, 2*i/theta.size))
    txt.set_alpha(max(0,min(1, 2*(i-400)/theta.size)))
    
    x.append(ts[i])
    y.append(theta[i])
    
    trace.set_data(x, y)
    #ax.scatter(T,TH, s=5, color = 'b')
    
    
    
ani = animation.FuncAnimation(fig, animate, interval=30, frames=theta.size, repeat = False)

#ani.save(path + 'quad.gif', writer='pillow', fps=30, dpi=100)
#%% Animate phase diagram

x = []
y = []
rg = 1.2

thmax = np.max(theta)
wmax =np.max(omega)

fig, ax = plt.subplots(figsize=(12,16))

ax.scatter(th, np.diff(th, append=0)/np.diff(t, append=0), color = 'g', s=5, label = 'Measured Data')

trace, = ax.plot([], [], lw = 2, color = 'b', label = 'Model')

#ax.set_xlim( -rg*thmax*0.8, rg*thmax)
#ax.set_ylim(-rg*wmax*1.2, rg*wmax)
ax.set_xlabel('$\\theta$ [Degrees]', fontsize = 20)
ax.set_ylabel("$\\omega$ [Degrees/s]", fontsize = 20)
ax.set_title('Phase Diagram', fontsize = 20)

ax.legend(fontsize = 15)

S = 10
def animate(i):
    
    trace.set_data(theta[:S*i], omega[:S*i])
    
ani = animation.FuncAnimation(fig, animate, interval=33.33, frames=int(theta.size/S), repeat = False)

#ani.save(path + 'Phase_Diagram.gif', writer='pillow', fps=30, dpi=100)

