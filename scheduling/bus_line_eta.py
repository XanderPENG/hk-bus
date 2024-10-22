"""
Author: Xander Peng
Date: 2024/10/22
Description: 
"""
import logging
import numpy as np
from typing_extensions import deprecated
from scheduling.bus_route import BusRoute
from scheduling.eta import Eta

class BusLineEta:

    def __init__(self, co, line_id, dir, service_type, dest, rmk, etas):
        # static
        self.co = co
        self.line_id = line_id
        self.dir = dir
        self.service_type = service_type
        self.dest = dest
        self.rmk = rmk
        self.logger = logging.getLogger('BusLineEta')

        # dynamic
        self.etas = etas
        self.eta_groups = self.__initialize_eta_group__()
        self.num_stops = len(next(iter(self.eta_groups.values())).keys())
        self.all_time_intervals = {_ + 1: [] for _ in
                                   range(self.num_stops - 1)}  # {pair_seq: [time_intervals]}; pair_seq starts from 1
        self.routes = []  # List of identified BusRoute

    def __initialize_eta_group__(self):
        """
        Split the etas into groups based on their stop_seq and scrapped_time
        convert to a dictionary: {T: {seq: [etas]}}
        """
        eta_groups = {}
        eta_scrapped_times = set([eta.scrapped_time for eta in self.etas])
        # Create index order for the times
        scrapped_times = list(eta_scrapped_times)
        scrapped_times.sort()
        scrapped_times_index = {time: i for i, time in enumerate(scrapped_times)}

        for eta in self.etas:
            if scrapped_times_index.get(eta.scrapped_time) not in eta_groups:
                eta_groups[scrapped_times_index.get(eta.scrapped_time)] = {}
                eta_groups[scrapped_times_index.get(eta.scrapped_time)].update({eta.seq: [eta]})

            elif eta.seq not in eta_groups[scrapped_times_index.get(eta.scrapped_time)]:
                eta_groups[scrapped_times_index.get(eta.scrapped_time)].update({eta.seq: [eta]})

            else:
                eta_groups[scrapped_times_index.get(eta.scrapped_time)][eta.seq].append(eta)

        return eta_groups

    def backward_search_feasible_routes(self):
        """
        Search all the feasible routes of this bus line:
            1. Find the feasible etas in the previous stop within this scraped time;
                a. if there is not any feasible eta, then try to find the feasible etas in the previous stop within the previous scraped time (till get a feasible eta)
                b. if there are several possible etas, try to assess the feasibility of each eta (e.g., based on the mean_gap, or closet gap)
            2. Create the bus route instance
            3. Here, we can create a list to store and update the avg_gap of each stop pair,
                which can be used to judge the feasibility of the etas.

        feasibility assessment:
        General: compare the 3 etas in Seq, T with the 3 etas in Seq-1, T
            a. filter the etas greater than any of the 3 etas in Seq-1, T -> find the least variance option (or, find the least one in the previous group for the least one in the current group and then remove this option)
            b. keep the BusRoute on waiting (or step to T-1 for searching) if the feasible etas are not found
            c. remove selected etas from the eta_groups


        :return:
        """
        self.logger.info(f"Start searching feasible routes for {self.co}-{self.line_id}-{self.dir}-{self.dest}")
        # Create a copy of the eta group, search and remove the matched etas
        # self.eta_groups = self.eta_groups.copy()
        # When there is no more etas in the unmatched_etas, the while loop will be terminated
        while not self.is_eta_group_empty():
            self.logger.info(f"Remaining etas: {self.count_unmatched_etas()}")
            # Ensure matching the eta from the latest scrapped time (remaining) and the latest seq (remaining)
            unmatched_max_timeframe = max(list(
                map(lambda timeframe_group: timeframe_group[0] if BusLineEta.are_all_lists_empty(
                    timeframe_group[1]) is False else -1,
                    self.eta_groups.items())))
            unmatched_max_seq = max(list(map(lambda seq_group: seq_group[0] if len(seq_group[1]) > 0 else -1,
                                             self.eta_groups.get(unmatched_max_timeframe).items())))

            last_etas = self.eta_groups.get(unmatched_max_timeframe).get(unmatched_max_seq)
            last_etas = sorted(last_etas, key=lambda x: x.eta,
                               reverse=True)  # sort the etas by the eta in descending order
            for last_eta in last_etas:
                # Create a BusRoute instance
                bus_route = BusRoute(self.co, self.line_id, last_eta.dest, self.num_stops)

                current_timeframe = unmatched_max_timeframe
                current_seq = unmatched_max_seq

                self.backward_search_feasible_route(last_eta, current_timeframe, current_seq, bus_route)
                # Add the bus_route into the routes
                self.routes.append(bus_route)

        self.logger.info(f"Finish searching feasible routes for {self.co}-{self.line_id}-{self.dir}-{self.dest}")
        self.logger.info(f"{len(self.routes)} Routes have been identified, with"
                         f"{len(list(filter(lambda route: route.is_complete() is True, self.routes)))} complete routes and "
                         f"{len(list(filter(lambda route: route.is_complete() is False, self.routes)))} incomplete routes, respectively")

    def backward_search_feasible_route(self, current_eta: Eta, timeframe, seq, bus_route: BusRoute):
        still_searching = True
        current_seq = seq
        # Add the basic information of the current eta into the bus_route
        bus_route.stops[current_seq - 1] = current_eta.seq
        bus_route.etas[current_seq - 1] = current_eta
        bus_route.eta_idx[current_seq - 1] = current_eta.id
        # Remove the start eta from the unmatched_etas
        self.eta_groups.get(timeframe).get(current_seq).remove(current_eta)

        while still_searching:
            current_time_intervals = self.all_time_intervals.get(current_seq - 1)
            feasible_eta = self.find_feasible_eta(current_eta, timeframe, current_seq, current_time_intervals)
            if feasible_eta is not None:
                # update the bus_route
                bus_route.stops[current_seq - 2] = feasible_eta.seq
                bus_route.etas[current_seq - 2] = feasible_eta
                bus_route.eta_idx[current_seq - 2] = feasible_eta.id
                # update the all_time_intervals
                # time_interval = abs(current_eta.eta - feasible_eta.eta).total_seconds() / 60
                # self.all_time_intervals.get(current_seq-1).append(time_interval)
                # update the loop variables
                current_eta = feasible_eta
                current_seq -= 1
            else:
                still_searching = False

    def find_feasible_eta(self, current_eta: Eta, current_timeframe, current_seq,
                          time_intervals: list, error_threshold: float = 5) -> Eta or None:
        """
        Find the feasible eta by searching the previous groups (in different timeframes)

        """
        _current_timeframe = current_timeframe
        target_stop = current_seq - 1

        feasible_eta = None
        while _current_timeframe >= 0:
            if target_stop in self.eta_groups.get(_current_timeframe):
                prev_group = self.eta_groups.get(_current_timeframe).get(target_stop)
                candidate_eta = current_eta.find_closet_earlier_eta(prev_group)
                if candidate_eta is not None and len(time_intervals) != 0:
                    # Calculate the time interval between the two stops
                    time_interval = abs(current_eta.eta - candidate_eta.eta).total_seconds() / 60
                    avg_interval = np.mean(time_intervals)
                    if abs(time_interval - avg_interval) <= error_threshold:
                        feasible_eta = candidate_eta
                        break
                elif candidate_eta is not None and len(time_intervals) == 0:
                    feasible_eta = candidate_eta
                    break
                else:
                    pass
            _current_timeframe -= 1

        if feasible_eta is not None:
            # Remove the matched eta from the unmatched_etas
            self.eta_groups.get(_current_timeframe).get(target_stop).remove(feasible_eta)
            # Update the time_intervals
            time_intervals.append(abs(current_eta.eta - feasible_eta.eta).total_seconds() / 60)

        return feasible_eta

    @staticmethod
    def are_all_lists_empty(d):
        if isinstance(d, dict):
            return all(BusLineEta.are_all_lists_empty(v) for v in d.values())
        elif isinstance(d, list):
            return len(d) == 0
        else:
            raise ValueError(f"Expected dict or list, got {type(d)}")

    def is_eta_group_empty(self):
        """
        Check if the eta group is empty
        """
        return BusLineEta.are_all_lists_empty(self.eta_groups)

    def remove_empty_subgroups(self):
        """
        Remove the empty subgroups in the eta_groups
        """
        for key in self.eta_groups.keys():
            for sub_key in self.eta_groups.get(key).keys():
                if len(self.eta_groups.get(key).get(sub_key)) == 0:
                    self.eta_groups.get(key).pop(sub_key)

    def count_unmatched_etas(self):
        """
        Count the number of unmatched etas
        """
        count = 0
        for key in self.eta_groups.keys():
            for seq in self.eta_groups.get(key).keys():
                count += len(self.eta_groups.get(key).get(seq))

        return count

    @deprecated("This method is not used anymore")
    def clean_etas(self, time_threshold: float = 5):
        """
        Identify eta records with small-time difference (within the threshold; default 2 minutes),
        and only keep the one with the latest eta
        """
        keys = list(self.eta_groups.keys())
        total_seq = set([key[0] for key in keys])
        total_time = set([key[1] for key in keys])
        num_time = len(total_time)

        new_groups = {}
        # Compare each eta in the current group with the etas in the following groups until the time_diff increases
        for seq in total_seq:
            time_index = 0
            inner_group = []
            while time_index < num_time - 1:
                if time_index == 0:
                    current_group = self.eta_groups.get((seq, time_index))
                else:
                    current_group = inner_group
                next_group = self.eta_groups.get((seq, time_index + 1))
                if current_group and next_group:
                    inner_group = self.clean_consecutive_eta_group(current_group, next_group, time_threshold)
                time_index += 1
            new_groups[seq] = inner_group

        self.eta_groups = new_groups

    @deprecated("This method is not used anymore")
    def clean_consecutive_eta_group(self, eta_group1: dict, eta_group2: dict, time_threshold: float):
        """
        Compare and clean two consecutive groups of etas (group by stop_seq and scrapped_time);
        to ensure the principles of:
            1. retain the latest scrapped eta record if the time difference is within the threshold
            2. one-out-one-retain: only remove 'n' records in the first group if 'n' records identified within the threshold
                here, the judgement of which record should be removed needs to be
            3. remove the eta records (if applicable) in the first group
            4. add the retained eta records into a new group

        :return: a new group of etas
        """
        to_remove_eta = []
        same_eta_record = []

        for eta1 in eta_group1:
            for eta2 in eta_group2:
                if eta1.equals(eta2, time_threshold) and eta2 not in same_eta_record:
                    to_remove_eta.append(eta1)
                    same_eta_record.append(eta2)

        new_group = [eta for eta in eta_group1 if eta not in to_remove_eta]
        new_group.extend(eta_group2)

        return new_group

    @deprecated("This method is not used anymore")
    def find_possible_consecutive_eta_pairs(self, num=5):
        """
        Find the possible consecutive eta pairs, and build the tree
        """
        num_seqs = len(self.eta_groups)
        _seq = num_seqs  # Since the seq starts from 1, ...

        all_candidate_eta_lists = []
        all_candidate_eta_time_gaps = []
        while _seq > 1:  # Backward search
            current_group = self.eta_groups.get(_seq)
            previous_group = self.eta_groups.get(_seq - 1)

            candidate_eta_lists = []
            for eta in current_group:
                # Find the feasible etas in the previous stop
                feasible_etas = eta.backward_search_feasible_etas(previous_group, num)
                ## Add the feasible etas
                if feasible_etas:
                    candidate_eta_pairs = []
                    for feasible_eta in feasible_etas:
                        candidate_eta_pairs.append((eta, feasible_eta))
                    candidate_eta_lists.append(candidate_eta_pairs)

            # calculate the time gap between the eta pairs
            candidate_eta_time_gaps = []
            for candidate_eta_pairs in candidate_eta_lists:
                g = []
                for eta_pair in candidate_eta_pairs:
                    time_gap = abs(eta_pair[0].eta - eta_pair[1].eta).total_seconds() / 60
                    g.append(time_gap)
                candidate_eta_time_gaps.append(time_gap)

            all_candidate_eta_lists.append(candidate_eta_lists)
            all_candidate_eta_time_gaps.append(candidate_eta_time_gaps)
            # Find the combination with the smallest standard deviation of the time gap (Optimization)
            _seq -= 1

        return all_candidate_eta_lists, all_candidate_eta_time_gaps

        # '''
        # Clean the tree: remove the nodes that do not have children
        # '''
        # is_feasible = False
        # while not is_feasible:
        #     is_feasible = True
        #     for node in self.tree.all_nodes():
        #         if not self.tree.children(node.identifier):
        #             self.tree.remove_node(node.identifier)
        #             is_feasible = False

        def __optimize_eta_pair_combination__():
            pass