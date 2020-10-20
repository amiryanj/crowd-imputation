from followbot.scenarios.real_scenario import RealScenario
from followbot.util.basic_geometry import Line
import re


class HermesScenario(RealScenario):
    """
    Class for replaying Hermes (Bottleneck) Crowd Experiments
    """
    def __init__(self):
        super(HermesScenario, self).__init__()

    def setuup(self):
        super().setup()
        exp_dimensions = re.split('-|\.', annotation_file)[-4:-1]
        if '2D' in annotation_file:
            line_objs = corridor_map(int(exp_dimensions[0]) / 100., int(exp_dimensions[0]) / 100.)
        else:
            line_objs = corridor_map(int(exp_dimensions[1])/100., int(exp_dimensions[2])/100.)
        for line_obj in line_objs:
            self.world.add_obstacle(line_obj)

def corridor_map(width, bottleneck):
    wall_b = Line((-4, 0), (4, 0))
    wall_t = Line((-4, width), (4, width))
    bottleneck_b = Line((4, -1), (4, (width - bottleneck)/2.))
    bottleneck_t = Line((4, width+1), (4, (width + bottleneck) / 2.))
    stand_b = Line((-4, 0), (-4, -1))
    stand_t = Line((-4, width), (-4, width+1))
    lines = [wall_b, wall_t, bottleneck_b, bottleneck_t]
    return lines
