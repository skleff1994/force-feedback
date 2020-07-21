from math import cos, sin
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def animatePointMass(xs, sleep=50):
    print("processing the animation ... ")
    cart_size = 1.
    pole_length = 5.
    fig = plt.figure()
    ax = plt.axes(xlim=(-8, 8), ylim=(-6, 6))
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
        py = 0.
        vx = np.asscalar(xs[i][1])
        vy = 0
        patch.set_xy([px - cart_size / 2, 0])
        time = i * sleep / 1000.
        time_text.set_text('time = %.1f sec' % time)
        return patch, line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True)
    print("... processing done")
    plt.show()
    return anim


def plotPointMass(xs, us, dt=1e-2):
    # Extract and plot trajectories
    T = len(us)
    tspan = np.linspace(0,T*dt,T+1)
    x1 = np.zeros(len(xs)) # p
    x2 = np.zeros(len(xs)) # v
    x3 = np.zeros(len(xs)) # lmb
    for i in range(len(xs)):
        x1[i] = xs[i][0]
        x2[i] = xs[i][1]
        x3[i] = xs[i][2]
    u = np.zeros(len(us)) # u
    for i in range(len(us)):
        u[i] = us[i]
    import matplotlib.pyplot as plt
    # Position
    fig, ax = plt.subplots(4,1)
    ax[0].plot(tspan, x1, 'b-', linewidth=3, label='p')
    ax[0].set_title('Position p', size=16)
    ax[0].set(xlabel='time (s)', ylabel='p (m)')
    # Velocity
    ax[1].plot(tspan, x2, 'b-', linewidth=3, label='v')
    # ax[1].plot(tspan, X[:,1], 'g-', linewidth=3, label='p')
    ax[1].set_title('Velocity v', size=16)
    ax[1].set(xlabel='time (s)', ylabel='v (m/s)')
    # Contact
    ax[2].plot(tspan, x3, 'b-', linewidth=3, label='lambda')
    ax[2].set_title('Contact force lambda', size=16)
    ax[2].set(xlabel='time (s)', ylabel='lmb (N)')
    # Input force
    ax[3].plot(tspan[:T], u, 'k-', linewidth=3, label='u')
    ax[3].set_title('Input force u', size=16)
    ax[3].set(xlabel='time (s)', ylabel='u (N)')
    # Legend
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Point mass trajectory with [p0, v0] = [0,1] and u = 0', size=16)
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