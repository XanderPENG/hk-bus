from datetime import datetime
from treelib import Node, Tree

class Eta:
    def __init__(self, co, line_id, dir, service_type, seq, dest, eta_seq, eta, rmk, scrapped_time):
        assert isinstance(eta, datetime), f"Expected datetime, got {type(eta)}"
        assert isinstance(scrapped_time, datetime), f"Expected datetime, got {type(scrapped_time)}"
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

    def equals(self, other: 'Eta', time_threshold: float or int) -> bool:
        assert isinstance(other, Eta), f"Expected Eta, got {type(other)}"
        return (self.co == other.co
                and self.line_id == other.line_id
                and self.dir == other.dir
                and self.dest == other.dest
                and self.seq == other.seq
                and abs(self.eta-other.eta).total_seconds() / 60 <= time_threshold
                )



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
        ## Create a tree with the root node
        self.tree = Tree()
        self.tree.create_node(identifier=self.co+'_'+self.line_id)


    def __initialize_eta_group__(self):
        """
        Split the etas into groups based on their stop_seq and scrapped_time
        """
        eta_groups = {}
        for eta in self.etas:
            if (eta.seq,eta.scrapped_time) not in eta_groups:
                eta_groups[(eta.seq,eta.scrapped_time)] = []
            eta_groups[eta.seq].append(eta)

        return eta_groups



    def clean_etas(self, time_threshold: float or int):
        """
        Identify eta records with small-time difference (within the threshold),
        and only keep the one with the latest eta
        """
        for seq_time, etas in self.eta_groups.items():
            new_eta_group = []
            # Compare each eta in the current group with the etas in the following groups until the time_diff increases

    @staticmethod
    def clean_consecutive_eta_group(eta_group1: dict, eta_group2: dict, time_threshold: float or int):
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
        pass