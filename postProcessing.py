
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

import fym.logging as logging
from fym.utils import rot


def plot_rllib_test(dir_save, dir_save_env):
    env_data = logging.load(dir_save_env)

    time = env_data['t']
    action = env_data['action']
    pos = env_data['quad']['pos'].squeeze()
    vel = env_data['quad']['vel'].squeeze()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(pos[:,0], pos[:,1])
    ax.set_ylabel('Y [m]')
    ax.set_xlabel('X [m]')
    ax.set_title('Trajectory')
    ax.grid(True)
    fig.savefig(Path(dir_save, "trajectory.png"), bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(time, pos[:,0])
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[:,1])
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "position.png"), bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(time, vel[:,0])
    ax[0].set_ylabel("$V_x$ [m/s]")
    ax[0].set_title("Velocity")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, vel[:,1])
    ax[1].set_ylabel("$V_y$ [m/s]")
    ax[1].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "velocity.png"), bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(time, action[:,0])
    ax[0].set_ylabel("$a_x [m/s^2]$")
    ax[0].set_title("Acceleration")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, action[:,1])
    ax[1].set_ylabel("$a_y [m/s^2]$")
    ax[1].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "acceleration.png"), bbox_inches='tight')
    plt.close('all')
