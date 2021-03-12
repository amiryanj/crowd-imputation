import os
import numpy as np
from OpenTraj.tools.parser.parser_eth import ParserETH
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from OpenTraj.tools.parser.parser_gc import ParserGC
from OpenTraj.tools.parser.parser_hermes import ParserHermes


def norm(x):
    return np.linalg.norm(x)

def unit_vec(x):
    return x / (norm(x) + 1E-8)

def angle(x):
    return np.arctan2(x[1], x[0])


# Occupancy Maps
grid = [np.linspace(-4., 4., 65), np.linspace(-2., 2., 33)]
grid_size = (len(grid[0]), len(grid[1]))
grid_area = (grid[0][-1]-grid[0][0], grid[1][-1]-grid[1][0])

def discretize(x, y):
    return int(x/grid_area[0] * grid_size[0]+grid_size[0]/2), int(y/grid_area[1] * grid_size[1]+grid_size[1]/2)

def is_groupmate(id1,id2):
    return (id2 in groupmates[id1])


open_traj_path = '/home/cyrus/workspace2/OpenTraj'

datasets = {
        # 'GC': ParserGC(os.path.join(open_traj_path, 'GC/Annotation')),
        'ETH-Univ': ParserETH(os.path.join(open_traj_path, 'ETH/seq_eth/obsmat.txt'),
                              os.path.join(open_traj_path, 'ETH/seq_eth/groups.txt')),
        'ETH-Hotel': ParserETH(os.path.join(open_traj_path, 'ETH/seq_hotel/obsmat.txt'),
                               os.path.join(open_traj_path, 'ETH/seq_hotel/groups.txt')),
        # #
        '1D-050-180-180': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-050-180-180_combined_MB.txt')),
        '1D-100-180-180': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-100-180-180_combined_MB.txt')),
        '1D-065-240-240': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-240-240-240_combined_MB.txt')),
        '1D-095-240-240': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-1D/uo-095-240-240_combined_MB.txt')),

        # 2D - Symmetric
        '2D-symmetric-300-050-050': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/bot-300-050-050_combined_MB.txt')),
        '2D-symmetric-300-100-100': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/bot-300-100-100_combined_MB.txt')),
        # 2D - Asymmetric
        '2D-asymmetric-300-050-085': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/boa-300-050-085_combined_MB.txt')),
        '2D-asymmetric-300-065-105': ParserHermes(os.path.join(open_traj_path, 'HERMES/Corridor-2D/boa-300-065-105_combined_MB.txt')),
    }

for name, dataset in datasets.items():
    print('dataset [%s] num peds = %d' % (name, len(dataset.id_t_dict)))
    groupmates = dataset.groupmates

    n_data = len(dataset.get_all_points())
    om_data = np.zeros((n_data, grid_size[0], grid_size[1]), dtype=np.float)

    counter = -1
    for t, ids_t in sorted(dataset.t_id_dict.items()):
        ps_t = dataset.t_p_dict[t]
        # vs_t = dataset.t_v_dict[t]
        for ii, id_i in enumerate(ids_t):
            counter += 1

            poi_pos = np.array(ps_t[ii])
            # poi_vel = np.array(vs_t[ii])
            # cener on i's person
            ps_t_centered = np.array(ps_t) - poi_pos

            # rotate
            poi_end_pos = dataset.id_p_dict[id_i][-1]
            # poi_dir = unit_vec(poi_end_pos - poi_pos)
            poi_angle = angle(poi_end_pos - poi_pos)
            poi_rot_mat = np.array([[np.cos(poi_angle), np.sin(poi_angle)],
                                    [-np.sin(poi_angle), np.cos(poi_angle)]],
                                   dtype=np.float)
            ps_t_rotated = np.matmul(poi_rot_mat, ps_t_centered.T).T

            # discretize & crop
            for jj, pos_j in enumerate(ps_t_rotated):
                id_j = ids_t[jj]
                if ii == jj: continue
                if not is_groupmate(id_i, id_j): continue
                u, v = discretize(pos_j[0], pos_j[1])
                # print(u, v)
                if 0 <= u < grid_size[0] and 0 <= v < grid_size[1]:
                    om_data[counter, u, v] = 1

    pom = om_data.mean(axis=0)
    # pom[int(grid_size[0] // 2), int(grid_size[1] // 2) + 10] = 0.1  # TEST

    plt.imshow(pom,  extent=[grid[1][0], grid[1][-1], grid[0][0], grid[0][-1]], origin='upper')
    plt.xlabel('laterl axis')
    plt.ylabel('x axis')
    plt.title('Dataset: %s' % name)
    plt.savefig(os.path.join('/home/cyrus/Dropbox/FollowBot/figs', 'acpom-%s-groupmates.png' % name))
    # plt.show()
    plt.clf()
    plt.close()
