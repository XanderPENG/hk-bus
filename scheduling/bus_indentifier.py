from datetime import datetime


from treelib import Node, Tree
from typing_extensions import deprecated


class Eta:
    def __init__(self, id, co, line_id, dir, service_type, seq, dest, eta_seq, eta, rmk, scrapped_time):
        assert isinstance(eta, datetime), f"Expected datetime, got {type(eta)}"
        assert isinstance(scrapped_time, datetime), f"Expected datetime, got {type(scrapped_time)}"
        self.id = id
        self.co = co
        self.line_id = line_id
        self.dir = dir
        self.service_type = service_type
        self.seq = seq
        self.dest = dest
        self.eta_seq = eta_seq
        self.eta = eta
        self.rmk = rmk
        self.scrapped_time = scrapped_time

    def equals(self, other, time_threshold: float) -> bool:
        assert isinstance(other, Eta), f"Expected Eta, got {type(other)}"
        return (self.co == other.co
                and self.line_id == other.line_id
                and self.dir == other.dir
                and self.dest == other.dest
                and self.seq == other.seq
                and abs(self.eta-other.eta).total_seconds() / 60 <= time_threshold
                )

    def find_closet_earlier_etas(self, others: list, num=1):
        """
        Search n closet etas in the previous stops
        """
        feasible_etas = filter(lambda x: x.eta < self.eta, others)
        feasible_etas = sorted(feasible_etas, key=lambda x: x.eta, reverse=True)
        return feasible_etas[:num]



class BusRoute:
    def __init__(self, co, line_id, dest,):
        self.co = co
        self.line_id = line_id
        self.dest = dest

        self.stops = []
        self.etas = []
        self.eta_idx = []

    def complete_route_by_avg_etas(self, avg_etas:list):
        """
        Complete the route based on the average etas
        """
        pass

class BusLineEta:
    def __init__(self, co, line_id, dir, service_type, dest, rmk, etas):
        # static
        self.co = co
        self.line_id = line_id
        self.dir = dir
        self.service_type = service_type
        self.dest = dest
        self.rmk = rmk

        # dynamic
        self.etas = etas
        self.eta_groups = self.__initialize_eta_group__()
        self.routes = []  # List of identified BusRoute
        ## Create a tree with the root node
        self.tree = Tree()
        self.tree.create_node(identifier=self.co+'_'+self.line_id)


    def __initialize_eta_group__(self):
        """
        Split the etas into groups based on their stop_seq and scrapped_time
        TODO: convert to a dictionary: {T: {seq: [etas]}}
        """
        eta_groups = {}
        eta_scrapped_times = set([eta.scrapped_time for eta in self.etas])
        # Create index order for the times
        scrapped_times = list(eta_scrapped_times)
        scrapped_times.sort()
        scrapped_times_index = {time: i for i, time in enumerate(scrapped_times)}

        for eta in self.etas:
            if (eta.seq,scrapped_times_index.get(eta.scrapped_time)) not in eta_groups:
                eta_groups[(eta.seq,scrapped_times_index.get(eta.scrapped_time))] = []
            eta_groups[(eta.seq, scrapped_times_index.get(eta.scrapped_time)) ].append(eta)

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
        pass



    @deprecated
    def clean_etas(self, time_threshold: float=5):
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
            while time_index < num_time-1:
                if time_index == 0:
                    current_group = self.eta_groups.get((seq, time_index))
                else:
                    current_group = inner_group
                next_group = self.eta_groups.get((seq, time_index+1))
                if current_group and next_group:
                    inner_group = self.clean_consecutive_eta_group(current_group, next_group, time_threshold)
                time_index += 1
            new_groups[seq] = inner_group

        self.eta_groups = new_groups

    @staticmethod
    def clean_consecutive_eta_group(eta_group1: dict, eta_group2: dict, time_threshold: float):
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

    @deprecated
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
            previous_group = self.eta_groups.get(_seq-1)

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
                