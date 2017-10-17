from abc import ABC
from typing import Tuple, List


class AbstractCell(ABC):
    def __init__(self,
                 center: Tuple[int, int]=(0, 0),
                 radius: int=0,
                 area: float=0.0,
                 velocity: Tuple[int, int]=(0, 0),
                 accel: Tuple[int, int]=(0, 0)):
        self._center: Tuple[int, int] = center
        self._radius: int = radius
        self._area: float = area
        self._velocity: Tuple[int, int] = velocity
        self._accel: Tuple[int, int] = accel

    def update(self, center: Tuple[float], velocity: Tuple[float, float]):
        self._center = center
        self._velocity = velocity

    def get_center(self):
        return self._center

    def get_radius(self):
        return self._radius

    def get_area(self):
        return self._area

    def get_velocity(self):
        return self._velocity

    def get_acceleration(self):
        return self._accel


class Cell(AbstractCell):
    def __init__(self,
                 center: Tuple[int, int]=(0, 0),
                 radius: int=0,
                 area: float=0.0,
                 velocity: Tuple[int, int]=(0, 0)):
        super(Cell, self).__init__(center=center, radius=radius, area=area, velocity=velocity)
        self._particles: List[CellParticle] = []

    def get_particles(self):
        return self._particles


class CellParticle(AbstractCell):
    def __init__(self,
                 center: Tuple[int, int]=(0, 0),
                 radius: int = 0,
                 area: float=0.0,
                 velocity: Tuple[int, int]=(0, 0),
                 weight: float=0.0):
        super(CellParticle, self).__init__(center=center, radius=radius, area=area, velocity=velocity)
        self._weight = weight

    def update_weight(self, weight):
        self._weight = weight

    def get_weight(self):
        return self._weight
