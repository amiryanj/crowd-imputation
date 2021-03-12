import numpy as np
from scipy.spatial.transform import Rotation


class Transform:
    def __init__(self, trans_=None, rot_=None):
        self.translation = trans_
        self.rotation = rot_

    def apply(self, pos, rot):
        out_orien = (self.rotation * rot).as_rotvec()
        out_trans = self.translation + self.rotation.apply(pos)
        return out_trans, out_orien

    def inverse(self, pos, rot):
        out_orien = (self.rotation.inv() * rot).as_rotvec()
        out_trans = self.rotation.inv().apply(pos - self.translation)
        return out_trans, out_orien


# test
if __name__ == "__main__":
    orien = Rotation.from_euler('z', np.pi/8, degrees=False)
    pos = np.array([2, 4, 0])

    t = Transform(np.array([3, 6, 5]), Rotation.from_euler('z', np.pi/8))
    print(pos, orien.as_rotvec())
    pos_tf, orien_tf = t.apply(pos, orien)
    print(pos_tf, orien_tf)
    pos_tf_tf, orien_tf_tf = t.inverse(pos_tf, Rotation.from_rotvec(orien_tf))
    print(pos_tf_tf, orien_tf_tf)
