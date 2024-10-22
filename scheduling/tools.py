"""
Author: Xander Peng
Date: 2024/10/22
Description: 
"""
import copy
import logging
import os.path
from typing import List

import pandas as pd

from scheduling.bus_route import BusRoute
from scheduling.eta import Eta


def find_and_merge_duplicates(complete_routes):
    duplicated_complete_routes_mapping = []

    # find duplicates
    for i, route in enumerate(complete_routes):
        mapping = []
        for j, route_ in enumerate(complete_routes):
            if i != j and route.equals(route_, 2):  # 如果i != j并且route相同
                mapping.append(j)
        if mapping:
            mapping.append(i)
            duplicated_complete_routes_mapping.append(mapping)

    # merge lists
    def merge_lists(nested_list):
        merged = []
        for sublist in nested_list:
            merged_with_existing = False
            for merged_sublist in merged:
                if set(sublist).intersection(merged_sublist):
                    merged_sublist.update(sublist)
                    merged_with_existing = True
                    break
            if not merged_with_existing:
                merged.append(set(sublist))
        return [list(merged_sublist) for merged_sublist in merged]

    len_mapping = len(duplicated_complete_routes_mapping)
    merging = True
    while merging:
        merged_mapping = merge_lists(duplicated_complete_routes_mapping)
        if len(merged_mapping) == len_mapping:
            merging = False
        else:
            len_mapping = len(merged_mapping)
            duplicated_complete_routes_mapping = merged_mapping

    # remove duplicates and keep one
    remove_route_indices = []
    for mapping in merged_mapping:
        keep_route_idx = mapping.pop()
        remove_route_indices.extend(mapping)

    # remove routes
    remove_routes = [complete_routes[idx] for idx in remove_route_indices]
    for route in remove_routes:
        complete_routes.remove(route)

    return complete_routes, merged_mapping


def find_mergeable_groups(incomplete_routes, all_intervals):
    mergeable_routes_mapping = []

    # find mergeable groups
    for i, route in enumerate(incomplete_routes):
        mapping = []
        for j, route_ in enumerate(incomplete_routes):
            if i != j and route.mergeable(route_, all_intervals, 2):  # 如果i != j并且route相同
                mapping.append(j)
        if mapping:
            mapping.append(i)
            mergeable_routes_mapping.append(mapping)

    # Sort the inner lists and remove duplicates
    mergeable_routes_mapping = [sorted(list(set(mapping))) for mapping in mergeable_routes_mapping]
    mergeable_routes_mapping = list(set([tuple(mapping) for mapping in mergeable_routes_mapping]))

    return mergeable_routes_mapping


def merge_all_routes(routes: list, all_intervals):
    # find mergeable groups
    mergeable_groups = find_mergeable_groups(routes, all_intervals)
    if not mergeable_groups:
        # no mergeable groups
        return routes
    else:
        # merge routes
        update_routes = []
        remove_routes_idx = []
        for group in mergeable_groups:
            candidate_routes = [routes[idx] for idx in group]  # Get the routes instances to be merged
            merged_route = copy.deepcopy(candidate_routes[0])
            # Merge the routes one by one
            for route in candidate_routes[1:]:
                merged_route.merge_routes(route)
            # Add the merged route to the update_routes list
            update_routes.append(merged_route)

            # record the index of the routes to be removed
            for route_idx in group:
                remove_routes_idx.append(route_idx)

        # remove merged routes
        remaining_routes = [route_ for idx, route_ in enumerate(routes) if idx not in remove_routes_idx]
        logging.warning(len(remaining_routes))
        logging.warning(len(update_routes))
        # Add remaining routes to the update_routes list
        update_routes.extend(remaining_routes)
        logging.info("iteration segment")
    return merge_all_routes(update_routes, all_intervals)


def output_complete_routes(routes: List[BusRoute], output_dir: str):
    full_df = pd.DataFrame()
    for i, route in enumerate(routes):
        # Attributes of the route
        co = route.co
        line_id = route.line_id
        dest = route.dest
        etas: List[Eta] = route.etas
        eta_idx: List[int] = route.eta_idx

        # Add the attributes to the dataframe
        num_rows = len(etas)
        single_df = pd.DataFrame(data={'route_id': [i] * num_rows,
                                       "co": [co] * num_rows,
                                       "line_id": [line_id] * num_rows,
                                       "dest": [dest] * num_rows,
                                       "seq": [eta.seq for eta in etas],
                                       "eta": [eta.eta for eta in etas],
                                       "scrapped_time": [eta.scrapped_time for eta in etas]
                                       })
        full_df = pd.concat([full_df, single_df], ignore_index=True)

    if output_dir:
        full_df.to_csv(output_dir, encoding='utf-8-sig')

    return full_df
