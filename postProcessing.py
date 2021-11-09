import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from collections import deque
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

import fym.logging as logging
from fym.utils import rot

def set_size(width_pt=245.7, fraction=1, subplots=(1,1)):
# def set_size(width_pt=155, fraction=1, subplots=(1,1)):
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27

    # golden_ratio = (5**.5 - 1) / 2
    golden_ratio = 3 / 4

    fig_width_in = fig_width_pt * inches_per_pt
    # fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_height_in = fig_width_in * golden_ratio
    return (fig_width_in, fig_height_in)
    
tex_fonts = {
    "text.usetex": True,
    "font.family": 'Times New Roman',
    "axes.labelsize": 6,
    "font.size": 6,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.grid": True,
}
plt.rcParams.update(tex_fonts)

def plot_rllib_test(dir_save, data_path):
    env_data = logging.load(data_path)

    time = env_data['t']
    action = env_data['action']
    pos = env_data['plant']['pos'].squeeze()
    vel = env_data['plant']['vel'].squeeze()
    lyap_dot = env_data['lyap_dot']
    width = 140

    fig, ax = plt.subplots(1, 1, figsize=set_size(width_pt=width))
    ax.plot(pos[:,0], pos[:,1], 'r')
    ax.set_ylabel('Y [m]')
    ax.set_xlabel('X [m]')
    ax.set_title('Trajectory')
    fig.savefig(Path(dir_save, "trajectory.pdf"), bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    ax[0].plot(time, pos[:,0], 'r')
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[:,1], 'r')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "position.pdf"), bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=155, subplots=(2,1)))
    ax[0].plot(time, vel[:,0], 'r')
    ax[0].set_ylabel("$V_x$ [m/s]")
    ax[0].set_title("Velocity")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, vel[:,1], 'r')
    ax[1].set_ylabel("$V_y$ [m/s]")
    ax[1].set_xlabel("time [s]")
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "velocity.pdf"), bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    ax[0].plot(time, action[:,0])
    ax[0].set_ylabel("$a_x [m/s^2]$")
    ax[0].set_title("Acceleration")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, action[:,1], 'r')
    ax[1].set_ylabel("$a_y [m/s^2]$")
    ax[1].set_xlabel("time [s]")
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "acceleration.pdf"), bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=set_size(width_pt=width))
    ax.plot(time, lyap_dot, 'r')
    ax.set_ylabel(r'$\frac{dV}{dt}$')
    ax.set_xlabel('time [s]')
    ax.set_title('Time derivative of Lyapunov candidate function')
    fig.savefig(Path(dir_save, "lyap_dot.pdf"), bbox_inches='tight')
    plt.close('all')


def plot_compare():
    dir_save = Path('./ray_results/compare/')

    # Compare btw L1 and L2
    pathes = [Path('./ray_results/PPO_2021-10-27_12-10-21/PPO_MyEnv_6bee6_00000_0_2021-10-27_12-10-21/checkpoint_001000/test_5/env_data.h5'),
              Path('./ray_results/PPO_2021-10-27_16-10-08/PPO_MyEnv_eb2e0_00000_0_2021-10-27_16-10-08/checkpoint_001000/test_5/env_data.h5')
              ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]
    width = 170

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time, pos[0][:,0], 'r')
    line2, = ax[0].plot(time, pos[1][:,0], 'b--')
    ax[0].legend(handles=(line1, line2),
                 labels=('L-reward No.1', 'L-reward No.2'))
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[0][:,1], 'r', time, pos[1][:,1], 'b--')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "L1-2.pdf"), bbox_inches='tight')
    plt.close('all')

    # Compare btw L3 and L4
    pathes = [Path('./ray_results/PPO_2021-10-27_18-06-35/PPO_MyEnv_2fb91_00000_0_2021-10-27_18-06-35/checkpoint_001000/test_5/env_data.h5'),
              Path('./ray_results/PPO_2021-10-27_19-45-57/PPO_MyEnv_112a5_00000_0_2021-10-27_19-45-57/checkpoint_001000/test_5/env_data.h5')
              ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time, pos[0][:,0], 'r')
    line2, = ax[0].plot(time, pos[1][:,0], 'b--')
    ax[0].legend(handles=(line1, line2),
                 labels=('L-reward No.3', 'L-reward No.4'))
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[0][:,1], 'r', time, pos[1][:,1], 'b--')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "L3-4.pdf"), bbox_inches='tight')
    plt.close('all')

    # Compare btw L1 and EQL1
    pathes = [Path('./ray_results/PPO_2021-10-27_12-10-21/PPO_MyEnv_6bee6_00000_0_2021-10-27_12-10-21/checkpoint_001000/test_5/env_data.h5'),
              Path('./ray_results/PPO_2021-10-22_15-20-16/PPO_MyEnv_1f7a9_00000_0_2021-10-22_15-20-16/checkpoint_001000/test_5/env_data.h5')
              ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time, pos[0][:,0], 'r')
    line2, = ax[0].plot(time, pos[1][:,0], 'b--')
    ax[0].legend(handles=(line1, line2),
                 labels=('L-reward No.1', 'EQL-reward No.1'))
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[0][:,1], 'r', time, pos[1][:,1], 'b--')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "L1-EQL1.pdf"), bbox_inches='tight')
    plt.close('all')

    # Compare btw L3 and EQL2
    pathes = [Path('./ray_results/PPO_2021-10-27_18-06-35/PPO_MyEnv_2fb91_00000_0_2021-10-27_18-06-35/checkpoint_001000/test_5/env_data.h5'),
              Path('./ray_results/PPO_2021-10-31_01-29-23/PPO_MyEnv_8a59d_00000_0_2021-10-31_01-29-23/checkpoint_001000/test_5/env_data.h5')
              ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time, pos[0][:,0], 'r')
    line2, = ax[0].plot(time, pos[1][:,0], 'b--')
    ax[0].legend(handles=(line1, line2),
                 labels=('L-reward No.3', 'EQL-reward No.2'))
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position")
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[0][:,1], 'r', time, pos[1][:,1], 'b--')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "L3-EQL2.pdf"), bbox_inches='tight')
    plt.close('all')


def plot_compare2():
    dir_save = Path('./ray_results/compare/')

    # Compare btw L2norm
    pathes = [
        Path('./ray_results/PPO_2021-10-30_11-14-17/PPO_MyEnv_15c55_00000_0_2021-10-30_11-14-17/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-30_04-39-15/PPO_MyEnv_e6a27_00000_0_2021-10-30_04-39-16/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-30_17-48-39/PPO_MyEnv_2d837_00000_0_2021-10-30_17-48-39/checkpoint_001000/test_5/env_data.h5')
    ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]
    width = 170

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time, pos[0][:,0], 'r')
    line2, = ax[0].plot(time, pos[1][:,0], 'b--')
    line3, = ax[0].plot(time, pos[2][:,0], 'g-.')
    ax[0].legend(handles=(line1, line2, line3),
                 labels=('N-reward No.1', 'N-reward No.2', 'N-reward No.3'))
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position using N-reward")
    ax[0].set_ylim([-1, 10])
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[0][:,1], 'r', time, pos[1][:,1], 'b--', time, pos[2][:,1], 'g-.')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylim([-0.5, 5.5])
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "N1-3.pdf"), bbox_inches='tight')
    plt.close('all')

    # Compare btw Q
    pathes = [
        Path('./ray_results/PPO_2021-10-29_00-16-02/PPO_MyEnv_f65c8_00000_0_2021-10-29_00-16-02/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-30_19-14-30/PPO_MyEnv_2b983_00000_0_2021-10-30_19-14-30/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-30_20-59-25/PPO_MyEnv_d4182_00000_0_2021-10-30_20-59-26/checkpoint_001000/test_5/env_data.h5')
    ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time, pos[0][:,0], 'r')
    line2, = ax[0].plot(time, pos[1][:,0], 'b--')
    line3, = ax[0].plot(time, pos[2][:,0], 'g-.')
    ax[0].legend(handles=(line1, line2, line3),
                 labels=('Q-reward No.1', 'Q-reward No.2', 'Q-reward No.3'))
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position using Q-reward")
    ax[0].set_ylim([-10, 70])
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[0][:,1], 'r', time, pos[1][:,1], 'b--', time, pos[2][:,1], 'g-.')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylim([-20, 50])
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "Q1-3.pdf"), bbox_inches='tight')
    plt.close('all')

    # Compare btw EQ
    pathes = [
        Path('./ray_results/PPO_2021-10-31_03-13-36/PPO_MyEnv_19739_00000_0_2021-10-31_03-13-36/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-30_07-49-32/PPO_MyEnv_7ba2e_00000_0_2021-10-30_07-49-32/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-31_08-35-38/PPO_MyEnv_16544_00000_0_2021-10-31_08-35-38/checkpoint_001000/test_5/env_data.h5')
    ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time, pos[0][:,0], 'r')
    line2, = ax[0].plot(time, pos[1][:,0], 'b--')
    line3, = ax[0].plot(time, pos[2][:,0], 'g-.')
    ax[0].legend(handles=(line1, line2, line3),
                 labels=('EQ-reward No.1', 'EQ-reward No.2', 'EQ-reward No.3'),
                 loc='upper center',
                 bbox_to_anchor=(0.42, -1.8),
                 ncol=3)
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position using EQ-reward")
    ax[0].set_ylim([-1, 8])
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time, pos[0][:,1], 'r', time, pos[1][:,1], 'b--', time, pos[2][:,1], 'g-.')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylim([-1, 8])
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "EQ1-3.pdf"), bbox_inches='tight')
    plt.close('all')

    # Compare btw N1, EQ2, L4
    pathes = [
        Path('./ray_results/PPO_2021-10-30_11-14-17/PPO_MyEnv_15c55_00000_0_2021-10-30_11-14-17/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-30_07-49-32/PPO_MyEnv_7ba2e_00000_0_2021-10-30_07-49-32/checkpoint_001000/test_5/env_data.h5'),
        Path('./ray_results/PPO_2021-10-27_19-45-57/PPO_MyEnv_112a5_00000_0_2021-10-27_19-45-57/checkpoint_001000/test_5/env_data.h5')
    ]

    env_data = [logging.load(path) for path in pathes]
    time = env_data[0]['t']
    action = [data['action'] for data in env_data]
    pos = [data['plant']['pos'].squeeze() for data in env_data]
    vel = [data['plant']['vel'].squeeze() for data in env_data]
    width=150

    fig, ax = plt.subplots(2, 1, figsize=set_size(width_pt=width, subplots=(2,1)))
    line1, = ax[0].plot(time[1000:], pos[0][1000:,0], 'r')
    line2, = ax[0].plot(time[1000:], pos[1][1000:,0], 'b--')
    line3, = ax[0].plot(time[1000:], pos[2][1000:,0], 'g-.')
    ax[0].legend(handles=(line1, line2, line3),
                 labels=('N-reward No.1', 'EQ-reward No.2', 'L-reward No.4'),
                 loc='upper center',
                 bbox_to_anchor=(0.5, -1.8),
                 ncol=3)
    ax[0].set_ylabel("X [m]")
    ax[0].set_title("Position")
    ax[0].set_ylim([-0.1, 0.3])
    ax[0].axes.xaxis.set_ticklabels([])
    ax[1].plot(time[1000:], pos[0][1000:,1], 'r', time[1000:], pos[1][1000:,1], 'b--', time[1000:], pos[2][1000:,1], 'g-.')
    ax[1].set_ylabel("Y [m]")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylim([-0.5, 0.5])
    [ax[i].grid(True) for i in range(2)]
    fig.align_ylabels(ax)
    fig.savefig(Path(dir_save, "N1-EQ2-L4.pdf"), bbox_inches='tight')
    plt.close('all')


if __name__ == "__main__":
    # plot_compare()
    plot_compare2()

