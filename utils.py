# Title : utils.py
# Author: Sebastien Kleff
# Date : 18.11.2020 
# Copyright LAAS-CNRS, NYU

# Utilities

from math import cos, sin
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

def animatePointMass(xs, sleep=1):
    '''
    Animate the point mass system with state trajectory xs
    '''
    # Check which model is used
    print(len(xs[0]))
    if(len(xs[0])>2):
        with_contact = True
    else:
        with_contact = False

    print("processing the animation ... ")
    cart_size = 1.
    fig = plt.figure()
    ax = plt.axes(xlim=(-7, 7), ylim=(-5, 5))
    patch = plt.Rectangle((0., 0.), cart_size, cart_size, fc='b')
    line, = ax.plot([], [], 'k-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        ax.add_patch(patch)
        line.set_data([], [])
        time_text.set_text('')
        return patch, line, time_text

    def animate(i):
        px = np.asscalar(xs[i][0])
        vx = np.asscalar(xs[i][1])
        patch.set_xy([px - cart_size / 2, 0])
        time = i * sleep / 1000.
        time_text.set_text('time = %.1f sec' % time)
        return patch, line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True)
    print("... processing done")
    plt.grid()
    plt.show()
    return anim


def plotPointMass(xs, us, dt=1e-2):
    '''
    Plots state-control trajectories  (xs,us) of point mass or point mass in contact
    '''
    # Check which model is used
    if(len(xs[0])>2):
        with_contact = True
    else:
        with_contact = False
    # Extract trajectories from croco sequences
    T = len(us)
    tspan = np.linspace(0,T*dt,T+1)
        # control traj
    u = np.zeros(len(us))  # control
    for i in range(len(us)):
        u[i] = us[i]
        # State traj
    x1 = np.zeros(len(xs)) # position
    x2 = np.zeros(len(xs)) # velocity
    for i in range(len(xs)):
        x1[i] = xs[i][0]
        x2[i] = xs[i][1]
        # Add force if using augmented model 
    if(with_contact):
        x3 = np.zeros(len(xs)) 
        for i in range(len(xs)):
            x3[i] = xs[i][2]
    # Create figs
    if(with_contact):
        fig, ax = plt.subplots(4,1)
    else:
        fig, ax = plt.subplots(3,1)
    # Plot position
    ax[0].plot(tspan, x1, 'b-', linewidth=3, label='p')
    ax[0].set_title('Position p', size=16)
    ax[0].set(xlabel='time (s)', ylabel='p (m)')
    ax[0].grid()
    # Plot velocity
    ax[1].plot(tspan, x2, 'b-', linewidth=3, label='v')
    ax[1].set_title('Velocity v', size=16)
    ax[1].set(xlabel='time (s)', ylabel='v (m/s)')
    ax[1].grid()
    # Plot force if necessary 
    if(with_contact):
        # Contact
        ax[2].plot(tspan, x3, 'b-', linewidth=3, label='lambda')
        ax[2].set_title('Contact force lambda', size=16)
        ax[2].set(xlabel='time (s)', ylabel='lmb (N)')
        ax[2].grid()
    # Plot control 
    ax[-1].plot(tspan[:T], u, 'k-', linewidth=3, label='u')
    ax[-1].set_title('Input force u', size=16)
    ax[-1].set(xlabel='time (s)', ylabel='u (N)')
    ax[-1].grid()
    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Point mass trajectory', size=16)
    plt.show()


def plotFiltered(Y_mea, X_hat, X_real, dt=1e-2):
    '''
    Plot point mass filtering using custom Kalman 
      Y_mea  : measurements
      X_hat  : estimates 
      X_real : ground truth
    '''
    # Extract trajectories and reshape
    T = len(Y_mea)
    ny = len(Y_mea[0])
    nx = len(X_real[0])
    tspan = np.linspace(0, T*dt, T+1)
    Y_mea = np.array(Y_mea).reshape((T, ny))
    X_hat = np.array(X_hat).reshape((T+1, nx))
    X_real = np.array(X_real).reshape((T+1, nx))
    # Create fig
    fig, ax = plt.subplots(2,1)
    # Plot position
    ax[0].plot(tspan[:T], Y_mea[:,0], 'b-', linewidth=2, alpha=.5, label='Measured')
    ax[0].plot(tspan, X_hat[:,0], 'r-', linewidth=3, alpha=.8, label='Filtered')
    ax[0].plot(tspan, X_real[:,0], 'k-.', linewidth=2, label='Ground truth')
    ax[0].set_title('Position p', size=16)
    ax[0].set(xlabel='time (s)', ylabel='p (m)')
    ax[0].grid()
    # Plot velocities
    ax[1].plot(tspan[:T], Y_mea[:,1], 'b-', linewidth=2, alpha=.5, label='Measured')
    ax[1].plot(tspan, X_hat[:,1], 'r-', linewidth=3, alpha=.8, label='Filtered')
    ax[1].plot(tspan, X_real[:,1], 'k-.', linewidth=2, label='Ground truth')
    ax[1].set_title('Velocities p', size=16)
    ax[1].set(xlabel='time (s)', ylabel='v (m/s)')
    ax[1].grid()
    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Kalman-filtered point mass trajectory', size=16)
    plt.show()

# def animateCartpole(xs, sleep=50):
#     print("processing the animation ... ")
#     cart_size = 1.
#     pole_length = 5.
#     fig = plt.figure()
#     ax = plt.axes(xlim=(-8, 8), ylim=(-6, 6))
#     patch = plt.Rectangle((0., 0.), cart_size, cart_size, fc='b')
#     line, = ax.plot([], [], 'k-', lw=2)
#     time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

#     def init():
#         ax.add_patch(patch)
#         line.set_data([], [])
#         time_text.set_text('')
#         return patch, line, time_text

#     def animate(i):
#         x_cart = np.asscalar(xs[i][0])
#         y_cart = 0.
#         theta = np.asscalar(xs[i][1])
#         patch.set_xy([x_cart - cart_size / 2, y_cart - cart_size / 2])
#         x_pole = np.cumsum([x_cart, -pole_length * sin(theta)])
#         y_pole = np.cumsum([y_cart, pole_length * cos(theta)])
#         line.set_data(x_pole, y_pole)
#         time = i * sleep / 1000.
#         time_text.set_text('time = %.1f sec' % time)
#         return patch, line, time_text

#     anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True)
#     print("... processing done")
#     plt.show()
#     return anim