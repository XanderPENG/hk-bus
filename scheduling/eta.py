"""
Author: Xander Peng
Date: 2024/10/22
Description: 
"""
from datetime import datetime
from typing import List

class Eta:
    def __init__(self, idx, co, line_id, direction, service_type, seq, dest, eta_seq, eta, rmk, scrapped_time):
        """
        Some params could be removed...while yet to be determined

        """
        assert isinstance(eta, datetime), f"Expected datetime, got {type(eta)}"
        assert isinstance(scrapped_time, datetime), f"Expected datetime, got {type(scrapped_time)}"
        self.id = idx
        self.co = co
        self.line_id = line_id
        self.dir = direction
        self.service_type = service_type
        self.seq = seq
        self.dest = dest
        self.eta_seq = eta_seq
        self.eta = eta
        self.rmk = rmk
        self.scrapped_time = scrapped_time

    def equals(self, other: 'Eta', time_threshold: float) -> bool:
        assert isinstance(other, Eta), f"Expected Eta, got {type(other)}"
        return (self.co == other.co
                and self.line_id == other.line_id
                and self.dir == other.dir
                and self.dest == other.dest
                and self.seq == other.seq
                and abs(self.eta-other.eta).total_seconds() / 60 <= time_threshold
                )

    def find_closet_earlier_eta(self, others: List['Eta']) -> 'Eta' or None:
        """
        Search the closet eta in the previous stops
        """
        feasible_etas = filter(lambda x: x.eta <= self.eta, others)
        feasible_etas = sorted(feasible_etas, key=lambda x: x.eta, reverse=True)
        if len(feasible_etas) == 0:
            return None
        return feasible_etas[0]