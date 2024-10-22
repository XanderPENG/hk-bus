"""
Author: Xander Peng
Date: 2024/10/22
Description: 
"""
import numpy as np
from datetime import datetime, timedelta
from scheduling.eta import Eta


class BusRoute:
    def __init__(self, co, line_id, dest, num_stops: int):
        self.co = co
        self.line_id = line_id
        self.dest = dest

        self.stops = []
        self.etas = []
        self.eta_idx = []
        self.__initialize__(num_stops)

    def __initialize__(self, num_stops: int):
        self.stops = [None] * num_stops
        self.etas = [None] * num_stops
        self.eta_idx = [None] * num_stops

    def is_complete(self):
        if None in self.stops and None in self.etas:
            return False
        return True

    def equals(self, other: 'BusRoute', time_threshold: float = 2, ratio_threshold=0.8) -> bool:
        """
        compare the two BusRoute instances and judge if they are the same, primarily based on the etas.
        """
        # assert isinstance(other, BusRoute), f"Expected BusRoute, got {type(other)}"
        similarity_list = []
        for idx, eta in enumerate(self.etas):
            if eta is not None and other.etas[idx] is not None:
                similarity_list.append(eta.equals(other.etas[idx], time_threshold))

        ratio = sum(similarity_list) / len(similarity_list)
        return ratio >= ratio_threshold

    def mergeable(self, other: 'BusRoute', avg_stop_pair_intervals: dict, time_threshold: float = 5) -> bool:
        """
        Check if the two incomplete BusRoute instances are mergeable.

        """
        # assert isinstance(other, BusRoute), f"Expected BusRoute, got {type(other)}"

        # Get the index list of non-None etas for two BusRoute instances
        self_eta_idx = [idx for idx, eta in enumerate(self.etas) if eta is not None]
        other_eta_idx = [idx for idx, eta in enumerate(other.etas) if eta is not None]

        if self_eta_idx == other_eta_idx:
            return False

        if len(self_eta_idx) > len(other_eta_idx):
            longer_route = self
            shorter_route = other
            longer_route_eta_idx = self_eta_idx
            shorter_route_eta_idx = other_eta_idx
        else:
            longer_route = other
            shorter_route = self
            longer_route_eta_idx = other_eta_idx
            shorter_route_eta_idx = self_eta_idx

        index = min(set(longer_route_eta_idx).difference(set(shorter_route_eta_idx)))

        # Judge if the index locating in the left or the right side of the shorter route
        if index < min(shorter_route_eta_idx):
            start_index = index
            end_index = min(shorter_route_eta_idx)
            start_eta = longer_route.etas[start_index]
            end_eta = shorter_route.etas[end_index]
        else:
            start_index = max(shorter_route_eta_idx)
            end_index = index
            start_eta = shorter_route.etas[start_index]
            end_eta = longer_route.etas[end_index]

        gap = end_index - start_index
        approx_interval = 0
        for i in range(gap):
            _interval = np.mean(avg_stop_pair_intervals.get(start_index+i+1))
            approx_interval += _interval
        real_interval = (end_eta.eta - start_eta.eta).total_seconds() / 60

        if abs(approx_interval - real_interval) <= time_threshold:
            return True
        return False

    def merge_routes(self, route2):
        """
        Merge two BusRoute instances;
        it can only be used for the mergeable BusRoute instances
        """
        # Get the index list of non-None etas for two BusRoute instances
        self_eta_idx = [idx for idx, eta in enumerate(self.etas) if eta is not None]
        other_eta_idx = [idx for idx, eta in enumerate(route2.etas) if eta is not None]

        num_stops = len(self.etas)
        merged_etas = []
        for i in range(num_stops):
            if i in self_eta_idx:
                merged_etas.append(self.etas[i])
            elif i in other_eta_idx:
                merged_etas.append(route2.etas[i])
            else:
                merged_etas.append(None)

        self.etas = merged_etas

    def complete_route_by_avg_etas(self, avg_stop_pair_intervals: dict):
        """
        Complete the route based on the average etas
        """
        num_stops = len(self.etas)
        self_eta_idx = [idx for idx, eta in enumerate(self.etas) if eta is not None]
        sep_idx = min(self_eta_idx)

        '''
        It is stupid to create a new Eta instance for each 'None' stop here... remains to be optimized
        '''

        # Forward completion
        for i in range(sep_idx+1, num_stops):
            if i not in self_eta_idx:
                self.etas[i] = Eta(None, self.co, self.line_id,
                                   self.etas[i-1].dir, self.etas[i-1].service_type,
                                   i, self.dest, None, self.etas[i-1].eta, None, self.etas[i-1].scrapped_time)
                self.etas[i].eta = self.etas[i-1].eta + timedelta(minutes=np.mean(avg_stop_pair_intervals.get(i)))
        # Backward completion
        for i in range(sep_idx-1, -1, -1):
            if i not in self_eta_idx:
                self.etas[i] = Eta(None, self.co, self.line_id,
                                      self.etas[i+1].dir, self.etas[i+1].service_type,
                                        i, self.dest, None, self.etas[i+1].eta, None, self.etas[i+1].scrapped_time)
                self.etas[i].eta = self.etas[i+1].eta - timedelta(minutes=np.mean(avg_stop_pair_intervals.get(i+1)))