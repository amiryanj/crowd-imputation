# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import os
import warnings
import collections
import numpy as np
import pandas as pd
import crowdrep_bot.crowd_sim.crowdsim as crowdsim
import crowdrep_bot.crowd_sim.umans_api as umans_api
from toolkit.loaders.loader_eth import load_eth

INDEFINITE_LOC = (-1000, -1000)


class TrajPredictor:
    """
    What we need:
        1. History of agents (obsv_traj)
        2. Context: (obstacles, ...)
        3.
    """
    def __init__(self, num_agents, obstacles, sim_model, backend="umans"):
        self.agent_radius = 0.3
        self.pref_speed = 1.2
        self.max_speed = 1.8
        self.max_acc = 1.0
        self.num_agents = num_agents

        if backend.lower() == "umans":
            self.sim = umans_api.CrowdSimUMANS(sim_model)
            for i in range(num_agents):
                self.sim.addAgent(INDEFINITE_LOC[0], INDEFINITE_LOC[1], self.agent_radius,
                                  prefSpeed=self.pref_speed, maxSpeed=self.max_speed, maxAcceleration=self.max_acc)
            if len(obstacles):
                warnings.warn("Obstacles can not be passed as arguments to UMANsS!")
        else:  # "crowdbag"
            self.sim = crowdsim.CrowdSim(sim_model.lower())
            self.sim.initSimulation(num_agents)
            for obs in obstacles:
                self.sim.addObstacleCoords(obs[0][0], obs[0][1], obs[1][0], obs[1][1])

        self.cur_agent_ids = []
        self.obsv_history = {}
        self.obstacles = obstacles.copy()

    def set_agents(self, agents_loc: dict, agents_vel: dict, agents_goal: dict):
        num_agents_t = len(agents_loc)

        # if simulator has more number of agents than what needed, set their loc to Indefinite
        for ii in range(num_agents_t, self.num_agents):
            self.sim.setPosition(ii, INDEFINITE_LOC[0], INDEFINITE_LOC[1])

        ii = -1
        for agent_id, loc in sorted(agents_loc.items()):  # iteration sorted on agent_id
            ii += 1

            # if simulator has less number of agents than what needed, add new agents
            if num_agents_t >= self.num_agents:
                try:
                    self.sim.addAgent(loc[0], loc[1], self.agent_radius, self.pref_speed, self.max_speed, self.max_acc)
                    self.num_agents += 1
                except Exception:
                    warnings.warn("Can not add new agents to CrowdBag during simulation")

            self.sim.setPosition(ii, loc[0], loc[1])
            self.sim.setVelocity(ii, agents_vel[agent_id][0], agents_vel[agent_id][1])
            self.sim.setGoal(ii, agents_goal[agent_id][0], agents_goal[agent_id][1])

            if agent_id not in self.obsv_history:
                self.obsv_history[agent_id] = []
            self.obsv_history[agent_id].append(loc)

        self.cur_agent_ids = sorted(agents_loc.keys())

    def predict(self, dts=[0.1]):
        pred_dps = np.zeros((len(dts), len(self.cur_agent_ids), 2))
        for t_index, dt in enumerate(dts):
            self.sim.doStep(dt)
            for i, id in enumerate(self.cur_agent_ids):
                dp_i = np.array(self.sim.getCenterVelocityNext(i)) * dt
                pred_dps[t_index, i, :] = pred_dps[t_index-1, i, :] + dp_i[:]
        pred_poss = pred_dps
        for i, id in enumerate(self.cur_agent_ids):
            pred_dps[:, i] += self.obsv_history[id][-1]
        return pred_poss


# test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")
    np.set_printoptions(precision=3)

    opentraj_root = "/home/cyrus/workspace2/OpenTraj"
    annot_file = os.path.join(opentraj_root, 'datasets/ETH/seq_eth/obsmat.txt')
    dataset = load_eth(annot_file, title="ETH-Univ")
    obstacles = [[[0, 0], [8, 0]]]
    frames = dataset.get_frames()
    all_agent_ids = dataset.data["agent_id"].unique()
    min_x, max_x = dataset.data["pos_x"].min() * 1.2, dataset.data["pos_x"].max() * 1.2
    min_y, max_y = dataset.data["pos_y"].min() * 1.2, dataset.data["pos_y"].max() * 1.2

    def last_loc(df: pd.DataFrame):
        return df[["pos_x", "pos_y"]].iloc[-1].to_numpy()

    agent_goals = dataset.data.groupby("agent_id").apply(last_loc)
    max_num_concurrent_agents = max([len(fr) for fr in frames])

    predictor_A = TrajPredictor(16, obstacles, "rvo2", "crowdbag")
    predictor_B = TrajPredictor(16, obstacles, "PowerLaw", "umans")

    _break = False
    _pause = False
    fig = plt.figure()
    def p(event):
        global _break, _pause
        if event.key == 'escape':
            _break = True
        if event.key == ' ':
            _pause = not _pause

    fig.canvas.mpl_connect('key_press_event', p)

    compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

    for t, fr in enumerate(frames):
        fr = fr.sort_values("agent_id")
        agent_ids = list(fr["agent_id"])
        poss_and_vels_t = fr[["pos_x", "pos_y", "vel_x", "vel_y"]].to_numpy()
        agent_goals_t = np.array([agent_goals[id] for id in agent_ids])

        # plot
        plt.cla()
        plt.title("Frame = %d" % t)
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])

        # make the predictions
        predictor_A.set_agents(dict(zip(agent_ids, poss_and_vels_t[:, :2])),
                               dict(zip(agent_ids, poss_and_vels_t[:, 2:4])),
                               dict(zip(agent_ids, agent_goals_t)))
        preds_A = predictor_A.predict([0.4, 0.4, 0.4, 0.4, 0.4])

        predictor_B.set_agents(dict(zip(agent_ids, poss_and_vels_t[:, :2])),
                               dict(zip(agent_ids, poss_and_vels_t[:, 2:4])),
                               dict(zip(agent_ids, agent_goals_t)))
        preds_B = predictor_B.predict([0.4, 0.4, 0.4, 0.4, 0.4])

        plt.plot(poss_and_vels_t[:, 0], poss_and_vels_t[:, 1], 'og')
        plt.plot(agent_goals_t[:, 0], agent_goals_t[:, 1], 'xr')

        # plt.plot(preds_sfm[:, :, 0], preds_sfm[:, :, 1], 'oc')

        for i in range(len(agent_ids)):
            pred_i_A = np.concatenate([poss_and_vels_t[i, :2].reshape(1, 2), preds_A[:, i].squeeze()])
            plt.plot(pred_i_A[:, 0], pred_i_A[:, 1], '--g')

            pred_i_B = np.concatenate([poss_and_vels_t[i, :2].reshape(1, 2), preds_B[:, i].squeeze()])
            plt.plot(pred_i_B[:, 0], pred_i_B[:, 1], '--m')

            # draw velocity vector
            plt.plot([poss_and_vels_t[i, 0], poss_and_vels_t[i, 0] + poss_and_vels_t[i, 2]],
                     [poss_and_vels_t[i, 1], poss_and_vels_t[i, 1] + poss_and_vels_t[i, 3]], 'y')
            plt.text(pred_i_A[0, 0], pred_i_A[0, 1], agent_ids[i], color="magenta")

        # plt.show()
        plt.pause(0.1)
        if _break:
            break
        if _pause:
            plt.pause(3)
