# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np


class FlowClass:
    def __init__(self, id, name, color, velocity):
        self.id = id
        self.name = name
        self.color = color
        self.velocity = velocity


class FlowClassifier:
    def __init__(self):
        self.classes = []
        self.preset_flow_classes = {0: FlowClass(0, 'still', 'g', velocity=np.zeros(2)),
                                    1: FlowClass(1, 'to_right', 'r', velocity=np.array([1.2, 0])),
                                    2: FlowClass(2, 'to_left', 'b', velocity=np.array([-1.2, 0])),
                                    }

    def classify(self, agents_locs, agents_vel):
        output = []
        if len(agents_vel):
            agents_orien = np.arctan2(agents_vel[:, 1], agents_vel[:, 0])

            for i in range(len(agents_orien)):
                # still agent
                if np.linalg.norm(agents_vel[i]) < 0.2:
                    output.append(self.preset_flow_classes[0])
                # towards_R agent
                elif -np.pi / 2 <= agents_orien[i] < np.pi / 2:
                    output.append(self.preset_flow_classes[1])
                # towards_L agent
                else:
                    output.append(self.preset_flow_classes[2])
        return output

    def id2color(self, ids):
        ids_np = np.array(ids).astype(int)
        if not ids_np.shape:
            return self.preset_flow_classes[ids_np].color
        ids_np_reshaped = ids_np.reshape(-1)
        colors = np.zeros_like(ids_np_reshaped, dtype='<U5')
        for i in range(ids_np.size):
            colors[i] = self.preset_flow_classes[ids_np_reshaped[i]].color
        return colors.reshape(ids_np.shape)
