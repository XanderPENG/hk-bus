"""
Author: Xander Peng
Date: 2024/8/19
Description: 
"""
import os

import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
from haversine import haversine_vector, Unit
import osmnx as ox
from typing import List, Dict, Tuple

from preprocessing.bus_line import Line


def getNodeInfo(nodesDict,
                nodeCoords,
                ):
    nodesDictKeys = list(nodesDict.keys())
    nodesDictValues = list(nodesDict.values())

    node_id = None
    if nodesDict == {}:
        return node_id

    # If the node is in the nodeDict
    if nodeCoords in nodesDictValues:
        node_id = nodesDictKeys[nodesDictValues.index(nodeCoords)]

    else:  # Find the 'same' node in the nodeDict (distance < 0.2m)
        dist_array = haversine_vector(np.array(nodesDictValues),
                                      np.array([nodeCoords]),
                                      Unit.METERS,
                                      comb=True)
        # Get the index of the 'same' node (if applicable)
        try:
            closeNode_idx = np.where(dist_array < 0.2)[1][0]
        except:
            closeNode_idx = None

        # If the point is close to an existing node, return the node id of the existing node
        if closeNode_idx is not None:
            node_id = nodesDictKeys[closeNode_idx]

    return node_id


def create_graph(road_gdf: gpd.GeoDataFrame,
                 length_col='length',
                 coords_col='coords'
                 ):
    splitG = nx.MultiDiGraph()

    # Create a dictionary to store the unique nodes (id, coords)
    nodesDict = {}

    start_nodes = []
    end_nodes = []

    node_idx = 0
    for idx, row in road_gdf.iterrows():
        # idx = row.index
        length = row[length_col]
        line = row['geometry']

        # (lat, lon) format
        lineStartCoords = (line.coords[0][1], line.coords[0][0])
        lineEndCoords = (line.coords[-1][1], line.coords[-1][0])

        startNode_id = getNodeInfo(nodesDict, lineStartCoords)
        if startNode_id is None:
            startNode_id = node_idx
            node_idx += 1
            nodesDict[startNode_id] = lineStartCoords

        endNode_id = getNodeInfo(nodesDict, lineEndCoords)
        if endNode_id is None:
            endNode_id = node_idx
            node_idx += 1
            nodesDict[endNode_id] = lineEndCoords

        start_nodes.append(startNode_id)
        end_nodes.append(endNode_id)

        # Add nodes to the graph
        if not splitG.has_node(startNode_id):
            splitG.add_node(startNode_id, x=lineStartCoords[1], y=lineStartCoords[0])
        if not splitG.has_node(endNode_id):
            splitG.add_node(endNode_id, x=lineEndCoords[1], y=lineEndCoords[0])
        splitG.add_edge(startNode_id, endNode_id,
                        length=length,
                        idx=idx)

    splitG.graph['crs'] = "EPSG:4326"

    indexed_road = road_gdf.copy()

    indexed_road['start_node'] = start_nodes
    indexed_road['end_node'] = end_nodes

    return splitG, indexed_road


def format_results(processed_lines: List[Line],
                   output_dir: str = None,
                   ) -> pd.DataFrame:
    result_dict = {}

    idx = 0
    for l in processed_lines:
        lineId = l.lineId
        lineName = l.get_name()
        lineDirection = l.get_direction()
        lineLength = l.get_length()
        lineStops = l.get_stopsList()
        lineCoords = l.get_stopsCoordsList()
        lineBusPairShortestPath = l.get_busPairShortestPath()[1:-1]
        result_dict[idx] = {'lineName': lineName,
                            'lineDirection': lineDirection,
                            'lineLength': lineLength,
                            'lineStops': lineStops,
                            'lineCoords': lineCoords,
                            'lineBusPairShortestPath': lineBusPairShortestPath}
        idx += 1

    result_df = pd.DataFrame(result_dict).T

    if output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except FileExistsError:
            pass

        result_df.to_csv(os.path.join(output_dir, 'matched_stops.csv'))

    return result_df


def getRoutePathLine(graph,
                     routePointCoordsList,
                     roadGdf):
    routePath = []
    routePathCoords = []
    routePathLineStrings = []
    for i in range(len(routePointCoordsList) - 2):
        startNodeCoord = routePointCoordsList[i]
        endNodeCoord = routePointCoordsList[i + 1]

        startNode = ox.distance.nearest_nodes(graph,
                                              startNodeCoord[0],
                                              startNodeCoord[1])
        endNode = ox.distance.nearest_nodes(graph,
                                            endNodeCoord[0],
                                            endNodeCoord[1])

        path = nx.shortest_path(graph,
                                source=startNode,
                                target=endNode,
                                weight='length')
        if i == 0:
            routePath.extend(path)
            routePathCoords.extend([graph.nodes[node]['x'], graph.nodes[node]['y']] for node in path)
        else:
            routePath.extend(path[1:])
            routePathCoords.extend([graph.nodes[node]['x'], graph.nodes[node]['y']] for node in path[1:])
        # routePath.extend(path)
        # routePathCoords.extend([graph.nodes[node]['x'], graph.nodes[node]['y']] for node in path)

    for j in range(len(routePath) - 2):
        startNode = routePath[j]
        endNode = routePath[j + 1]

        pathLine = roadGdf[(roadGdf['start_node'] == startNode) & (roadGdf['end_node'] == endNode)].geometry.iat[0]
        routePathLineStrings.append(pathLine)

    pathLineGdf = gpd.GeoDataFrame(geometry=routePathLineStrings, crs='EPSG:4326')
    return routePath, routePathCoords, routePathLineStrings, pathLineGdf
