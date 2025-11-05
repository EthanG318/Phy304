#3a
import numpy as np
import matplotlib.pyplot as plt
m=3.5 #kg
k= 1 # spring const
y0=0
v0=10
w=.5
w0=0.1
f0=3.2 #F Not
tf=300
pos_x=y0
vel_x=v0
wd=1
def DragDAMP(t,x,v):
    return -(b*v)-(k*x)+f0*np.cos(wd*t)
nsteps=2000
t,dt=np.linspace(0,tf,nsteps, retstep=True)
data2=np.zeros((nsteps,3))
for j, ti in enumerate(t):
    acc_x= DragDAMP(ti,pos_x,vel_x)/m
    vel_x+=acc_x*dt
    pos_x+=vel_x*dt
    data[j]=t[j],pos_x,vel_x #fill the data array with our data
plt.plot(data[:,0],data[:,1])
plt.xlabel("Time (s)")
plt.ylabel("Meters (m)")
plt.title("Harmonic Oscillator")
plt.show()
    