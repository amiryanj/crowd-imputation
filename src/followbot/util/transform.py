import numpy as np
from scipy.spatial.transform import Rotation


class Transform:
    def __init__(self, trans_=None, rot_=None):
        self.translation = trans_
        self.rotation = rot_

    def apply(self, pos, rot):
        out_orien = rot.__mul__(self.rotation).as_quat()
        out_trans = self.translation + self.rotation.apply(pos)
        return out_trans, out_orien
