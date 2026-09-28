from dataclasses import dataclass, asdict
import math
import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import Obstacle

@dataclass
class Scenario:
    name: str = 'baseline'
    vehicles: int = 6
    spacing: float = 45.0
    speed_spread: float = 0.0
    merge_length: float = 260.0
    traffic_speed: float = 22.0
    ego_speed: float = 20.0
    phase: float = 0.0
    rationale: str = 'Sparse traffic on two main lanes.'

    @classmethod
    def from_dict(cls, data):
        s = cls(**data)
        limits = {'vehicles': (2, 18), 'spacing': (18, 60),
                  'speed_spread': (0, 7), 'merge_length': (140, 300),
                  'traffic_speed': (16, 28), 'ego_speed': (16, 28), 'phase': (-15, 15)}
        if type(s.vehicles) is not int:
            raise ValueError('vehicles must be an integer')
        for key, (lo, hi) in limits.items():
            x = getattr(s, key)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not lo <= x <= hi:
                raise ValueError(f'{key} must be finite and in [{lo}, {hi}]')
        if not isinstance(s.name, str) or not s.name or len(s.name) > 100:
            raise ValueError('name must be 1..100 characters')
        if not isinstance(s.rationale, str) or len(s.rationale) > 3000:
            raise ValueError('rationale must be a string of at most 3000 characters')
        return s

    def complexity(self):
        # Explicit structural proxy, NOT a calibrated measure of driving difficulty.
        parts = [(self.vehicles-2)/16, (60-self.spacing)/42,
                 self.speed_spread/7, (300-self.merge_length)/160]
        return round(float(np.mean(parts)), 4)

class RampMergeEnv(AbstractEnv):
    """Ego starts in a straight acceleration lane, next to two main-road lanes.
    This deliberately omits the curved entry ramp; the task is gap selection/merge.
    Main traffic uses IDM longitudinal control with lane changes disabled.
    """
    def __init__(self, scenario=None, render_mode=None):
        self.scenario = Scenario.from_dict(scenario or asdict(Scenario()))
        super().__init__(render_mode=render_mode)

    @classmethod
    def default_config(cls):
        c = super().default_config()
        c.update(observation={'type': 'Kinematics', 'vehicles_count': 19,
                              'absolute': True, 'normalize': False},
                 action={'type': 'DiscreteMetaAction', 'target_speeds': [12,16,20,24,28,32]},
                 simulation_frequency=10, policy_frequency=2, duration=30,
                 screen_width=1000, screen_height=260, scaling=4,
                 centering_position=[0.25, 0.5],
                 neighbour_vehicles_connected_lanes=True)
        return c

    def _reset(self):
        s = self.scenario
        net = RoadNetwork()
        for lane in range(3):
            net.add_lane('a', 'b', StraightLane([-400, lane*4], [s.merge_length, lane*4],
                         line_types=[LineType.STRIPED, LineType.STRIPED]))
        for lane in range(2):
            net.add_lane('b', 'c', StraightLane([s.merge_length, lane*4], [s.merge_length+500, lane*4],
                         line_types=[LineType.STRIPED, LineType.STRIPED]))
        self.road = Road(network=net, np_random=self.np_random,
                         neighbour_vehicles_connected_lanes=True)
        self.road.objects.append(Obstacle(self.road, [s.merge_length-2, 8]))
        ego = self.action_type.vehicle_class(self.road, [40, 8], speed=s.ego_speed)
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)
        # Put relevant traffic both in front and behind ego, with seeded phase jitter.
        per_lane = math.ceil(s.vehicles/2)
        for i in range(s.vehicles):
            lane, row = i % 2, i // 2
            x = 40 + (row-(per_lane-1)/2)*s.spacing + s.phase + self.np_random.uniform(-2,2)
            speed = s.traffic_speed + self.np_random.uniform(-s.speed_spread,s.speed_spread)
            v = IDMVehicle(self.road, [x, lane*4], speed=speed, target_speed=speed,
                           enable_lane_change=False)
            self.road.vehicles.append(v)

    def success(self):
        v = self.vehicle
        return bool(not v.crashed and v.on_road and
                    v.position[0] >= self.scenario.merge_length+30 and
                    abs(v.position[1]-4*round(v.position[1]/4)) < 0.7 and
                    v.position[1] < 5.5)

    def _is_terminated(self):
        return bool(self.vehicle.crashed or not self.vehicle.on_road or self.success())

    def _is_truncated(self):
        return bool(self.time >= self.config['duration'])

    def _reward(self, action):
        if self.vehicle.crashed or not self.vehicle.on_road:
            return -1.0
        return 1.0 if self.success() else -0.01

    def _info(self, obs, action=None):
        info = super()._info(obs, action)
        info['success'] = self.success()
        info['on_road'] = bool(self.vehicle.on_road)
        return info


def rule_action(env):
    """Fixed full-state rule controller (privileged state, not an observation-only policy).
    No training. Uses gap checks for merging; performance is not a solvability oracle.
    """
    ego = env.vehicle
    # Continue lane change rather than repeatedly decrementing target lane.
    if abs(ego.position[1] - 4*ego.target_lane_index[2]) > 0.6:
        return 1  # IDLE
    target_y = 4 if ego.position[1] > 6 else 4*round(ego.position[1]/4)
    neighbours = [v for v in env.road.vehicles[1:] if abs(v.position[1]-target_y) < 1.5]
    front = min((v for v in neighbours if v.position[0] > ego.position[0]),
                key=lambda v:v.position[0], default=None)
    rear = max((v for v in neighbours if v.position[0] <= ego.position[0]),
               key=lambda v:v.position[0], default=None)
    front_gap = 1e6 if front is None else front.position[0]-ego.position[0]
    rear_gap = 1e6 if rear is None else ego.position[0]-rear.position[0]
    front_safe = front_gap > 8 + max(0,ego.speed-(front.speed if front else 0))*2
    rear_safe = rear_gap > 8 + max(0,(rear.speed if rear else 0)-ego.speed)*2
    if ego.position[1] > 6:
        if front_safe and rear_safe:
            return 0  # LANE_LEFT
        if not front_safe:
            return 4  # SLOWER
        return 3 if ego.speed < 28 else 1
    if front_gap < 8+ego.speed*0.8:
        return 4
    return 3 if ego.speed < env.scenario.traffic_speed-1 else 1
