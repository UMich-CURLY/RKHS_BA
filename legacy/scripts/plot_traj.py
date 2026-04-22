import sys, os
import numpy as np
import matplotlib.pyplot as plt
import ipdb
def plot_traj_all(xyz_all_traj,
                  name_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each curve
    for i in range(len(xyz_all_traj)):
        xyz = xyz_all_traj[i]
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], label = name_list[i][:-4])
        
    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def read_kitti_format(fnames):
    xyzs = []
    for i in fnames:
        pose_12 = np.genfromtxt(i)
        #xyz = pose_12[ np.idx_([j for j in range(pose_12.shape[0])], [3, 7, 11])]
        xyz = pose_12[:, np.asarray([3,7,11])]
        print(xyz.shape)
        xyzs.append(xyz)
    return xyzs

def read_tartan_format(fnames):
    xyzs = []
    for i in fnames:
        pose_12 = np.genfromtxt(i)

        xyz = pose_12[: ,:3]
        print(xyz.shape)
        xyzs.append(xyz)
    return xyzs

def read_tum_format(fnames):
    xyzs = []
    for i in fnames:
        pose_12 = np.genfromtxt(i)

        xyz = pose_12[: ,1:4]
        print(xyz.shape)
        xyzs.append(xyz)
    return xyzs

def read_general_format(fnames):
    xyzs = []
    for i in fnames:
        pose_12 = np.genfromtxt(i)        
        if i.endswith(".tum"):
            xyz = pose_12[: ,1:4]
        elif i.endswith(".tartan"):
            xyz = pose_12[:, :3]
        elif i.endwith(".kitti"):
            xyz = pose_12[:, np.asarray([3,7,11])]
        print(xyz.shape)
        xyzs.append(xyz)
    return xyzs



if __name__ == "__main__":

    traj_file_format = sys.argv[1]
    files = sys.argv[2:]

    num_traj = len(files)    
    if traj_file_format == "tartan":
        traj_all = read_tartan_format(files)
    elif traj_file_format == "kitti":
        traj_all = read_kitti_format(files)
    elif traj_file_format == "tum":
        traj_all = read_tum_format(files)
    else:
        traj_all = read_general_format(files)
    plot_traj_all(traj_all, files)
    
