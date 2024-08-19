"""
Author: Xander Peng
Date: 2024/8/19
Description: 
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# from shapely.ops import nearest_points
import networkx as nx
import osmnx as ox
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class Stop:
    def __init__(self,
                 stopName: str,
                 lon: float = None,
                 lat: float = None
                 ):
        self.stopName = stopName
        self.lon = lon
        self.lat = lat
        self.point = Point(lon, lat)

        self.finalLon = None
        self.finalLat = None

    def get_name(self):
        return self.stopName

    def get_coordinates(self):
        return (self.lon, self.lat)

    def set_coordinates(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def setFinalCoordinates(self, lon, lat):
        self.finalLon = lon
        self.finalLat = lat

    def findNearestLines(self,
                         bufferDist: float,  # Meter
                         networkGDF: gpd.GeoDataFrame
                         ):
        """Find the nearest lines to the stop within a buffer distance

        Args:
            bufferDist (float): Distance in meters

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing the lines intersecting with the buffer
        """
        stopBuffer = gpd.GeoDataFrame(geometry=[self.point.buffer(bufferDist / 111000)])
        stopBuffer = stopBuffer.set_crs('EPSG:4326')

        intersectedLines = gpd.overlay(networkGDF, stopBuffer, how='intersection')
        return intersectedLines

    def getProjectionPointsinNearestLines(self,
                                          bufferDist: float,  # Meter
                                          networkGDF: gpd.GeoDataFrame):
        """Get the projection points on the nearest lines to the stop within a buffer distance

        Args:
            bufferDist (float): Distance in meters

        Returns:
            CandidateProjPoints: A DataFrame containing the projection points (with tuple(lat, lon) in column '0')
            on the nearest lines
        """
        nearestLines = self.findNearestLines(bufferDist, networkGDF)
        candidateProjPoints = pd.DataFrame(
            nearestLines['geometry'].map(lambda x: x.interpolate(x.project(self.point)).coords[:][0]))

        candidateProjPoints['name'] = self.stopName
        candidateProjPoints = candidateProjPoints.rename(columns={0: 'coords'})
        return candidateProjPoints

    def matchStopNameAndCoords(self,
                               stopsShp: gpd.GeoDataFrame,
                               stopsShpNameCol: str,
                               preMatchedDict: dict,
                               ):
        """Match the stop name to the stops shapefile

        Args:
            stopsShp (gpd.GeoDataFrame): A GeoDataFrame containing the stops
            stopsShpNameCol (str): The column name of the stop name in the stops shapefile

        Returns:
            A list [(stopName1, lon, lat), (stopName2, lon, lat), ...]
        """
        if self.stopName in preMatchedDict.keys() and self.stopName in stopsShp[stopsShpNameCol].to_list():

            sliceShp = stopsShp[stopsShp[stopsShpNameCol] == self.stopName]
            matchedStopList = list(zip(sliceShp[stopsShpNameCol],
                                       [coords[0][0] for coords in sliceShp['geometry'].map(lambda x: x.coords)],
                                       [coords[0][1] for coords in sliceShp['geometry'].map(lambda x: x.coords)]))
            # matchedStopList = [preMatchedDict[self.stopName].get('Station-final')]

            ''' For Gov.data only below'''
        elif self.stopName in stopsShp[stopsShpNameCol].to_list():
            sliceShp = stopsShp[stopsShp[stopsShpNameCol] == self.stopName]
            matchedStopList = list(zip(sliceShp[stopsShpNameCol],
                                       [coords[0][0] for coords in sliceShp['geometry'].map(lambda x: x.coords)],
                                       [coords[0][1] for coords in sliceShp['geometry'].map(lambda x: x.coords)]))

        else:
            matchedStopInfoList = process.extract(self.stopName, stopsShp[stopsShpNameCol], scorer=fuzz.token_set_ratio,
                                                  limit=8  # Limit the number of matched stops
                                                  )
            # Filter those matched stops with name containing stopName, to narrow down the search space
            filteredMatchedStopInfoList = list(filter(lambda x: self.stopName in x[0], matchedStopInfoList))
            if len(filteredMatchedStopInfoList) == 0:
                pass
            else:
                matchedStopInfoList = filteredMatchedStopInfoList

            """TODO: 
            1. Filter those matched stops with name containing stopName, 
                to narrow down the search space - Done
            2. Identify the specific matched stop, and its coordinates, using the index returned by process.extract - Done
            3. lower down the dense of stopShp, by removing stops with the same name in the same line - Done
            """
            matchedStopNameList = [i[0] for i in matchedStopInfoList]
            matchedStopIndexList = [i[-1] for i in matchedStopInfoList]
            matchedStopCoordsList = stopsShp.loc[matchedStopIndexList]['geometry'].map(lambda x: x.coords[0]).to_list()
            matchedStopList = list(zip(matchedStopNameList,
                                       [coords[0] for coords in matchedStopCoordsList],
                                       [coords[1] for coords in matchedStopCoordsList]))
        return matchedStopList

    @DeprecationWarning
    @staticmethod
    def findStopCoordsFromShp(
            matchedStopList: gpd.GeoDataFrame,
            stopsShp: gpd.GeoDataFrame,
            stopsShpNameCol: str,
    ):
        matchedStopCoords = []
        for stopName in matchedStopList:
            # print(stopName)
            stopCoords = stopsShp[stopsShp[stopsShpNameCol] == stopName].geometry
            stopCoords = stopCoords.to_list()[0]
            stopCoords = (stopCoords.x, stopCoords.y)
            matchedStopCoords.append(stopCoords)
        return matchedStopCoords


class Line:
    def __init__(self,
                 lineId: int,
                 lineName: str,
                 lineDirection: str,
                 stopsNameList: list = None,
                 stopsCoordsList: list = None,
                 ):
        self.lineId = lineId
        self.lineName = lineName
        self.lineDirection = lineDirection
        self.stopsCoordsList = stopsCoordsList
        self.stopsNameList = stopsNameList
        self.length = None
        ''' Split the line into consective segments '''
        self.consectiveStopsName = [(self.stopsNameList[i], self.stopsNameList[i + 1])
                                    for i in range(len(self.stopsNameList) - 1)]
        self.findPathGraph = None
        self.busPairShortestPath = []

    def get_name(self):
        return self.lineName

    def get_direction(self):
        return self.lineDirection

    def get_stopsList(self):
        return self.stopsNameList

    def get_length(self):
        return self.length

    def get_stopsCoordsList(self):
        return self.stopsCoordsList

    def get_findPathGraph(self):
        return self.findPathGraph

    def get_busPairShortestPath(self):
        return self.busPairShortestPath

    def set_length(self, length):
        self.length = length

    def set_findPathGraph(self, findPathGraph):
        self.findPathGraph = findPathGraph

    def set_busPairShortestPath(self, busPairShortestPath):
        self.busPairShortestPath = busPairShortestPath

    def clearStopsNameList(self):
        self.stopsNameList = []

    def clearStopsCoordsList(self):
        self.stopsCoordsList = []

    def updateStopsInfo(self, newStopsNameList, newStopsCoordsList):
        self.stopsNameList = newStopsNameList
        self.stopsCoordsList = newStopsCoordsList

    @staticmethod
    def deriveStopName(
            stop: Stop,
            stopsShp: gpd.GeoDataFrame,
            stopsShpNameCol: str,
            preMatchedDict: dict,
    ):
        '''  @Deprecated
        if np.isnan(stop.lon) or stop.stopName not in stopsShp[stopsShpNameCol].to_list():
            matchedStopInfoList = stop.matchStopNameAndCoords(stopsShp, stopsShpNameCol, preMatchedDict)

            # matchedStopNameList = stop.matchStopName(stopsShp, stopsShpNameCol, preMatchedDict)
            # matchedStopCoordsList = Stop.findStopCoordsFromShp(matchedStopNameList, stopsShp, stopsShpNameCol)
            matchedStopNameList = [i[0] for i in matchedStopInfoList]
            matchedStopCoordsList = [(i[1], i[2]) for i in matchedStopInfoList]

        else:
            matchedStopNameList = [stop.stopName]
            matchedStopCoordsList = Stop.findStopCoordsFromShp(matchedStopNameList, stopsShp, stopsShpNameCol)
        '''

        matchedStopInfoList = stop.matchStopNameAndCoords(stopsShp, stopsShpNameCol, preMatchedDict)

        # matchedStopNameList = stop.matchStopName(stopsShp, stopsShpNameCol, preMatchedDict)
        # matchedStopCoordsList = Stop.findStopCoordsFromShp(matchedStopNameList, stopsShp, stopsShpNameCol)
        matchedStopNameList = [i[0] for i in matchedStopInfoList]
        matchedStopCoordsList = [(i[1], i[2]) for i in matchedStopInfoList]
        return matchedStopNameList, matchedStopCoordsList

    @staticmethod
    def findConsectivePath(fromStopInfo,  # tuple: (Stop, matchedStopNameList, matchedStopCoordsList)
                           #    thisStopInfo,
                           endStopInfo,
                           networkGraph: nx.Graph,
                           ):
        ''' preStopNearestNodes = [ox.distance.nearest_nodes(networkGraph, i[0], i[1])
                               for i in preStopInfo[2]]
        thisStopNearestNodes = [ox.distance.nearest_nodes(networkGraph, i[0], i[1])
                                for i in thisStopInfo[2]]
        nextStopNearestNodes = [ox.distance.nearest_nodes(networkGraph, i[0], i[1])
                                for i in nextStopInfo[2]]
        pathPre2This = nx.shortest_path(networkGraph, preStopNearestNodes[0], thisStopNearestNodes[0], weight='length')
        '''
        # print('raw:', fromStopInfo[0].get_name(), endStopInfo[0].get_name())
        # print('matched:', fromStopInfo[1], endStopInfo[1])
        fromStopNodes = [ox.distance.nearest_nodes(networkGraph, i[0], i[1])
                         for i in fromStopInfo[2]]
        endStopNodes = [ox.distance.nearest_nodes(networkGraph, i[0], i[1])
                        for i in endStopInfo[2]]
        pathFrom2End = []
        pathLengthFrom2End = []
        for i in range(len(fromStopNodes)):
            for j in range(len(endStopNodes)):
                try:
                    path = nx.shortest_path(networkGraph,
                                            fromStopNodes[i], endStopNodes[j],
                                            weight='length',
                                            method='dijkstra')
                    pathLength = nx.shortest_path_length(networkGraph,
                                                         fromStopNodes[i], endStopNodes[j],
                                                         weight='length',
                                                         method='dijkstra')
                except nx.NetworkXNoPath:
                    path = []
                    # Set a large number to indicate the path is not available
                    pathLength = 9999
                    # print('No path found between: ',
                    #       fromStopInfo[1][i], 'coords: ', fromStopNodes[2][i],
                    #       endStopInfo[1][j], 'coords: ', endStopNodes[2][j])
                pathFrom2End.append(path)
                pathLengthFrom2End.append(pathLength)
        # If pathLengthFrom2End is all 9999, then find the path using undirected graph
        if all([_ == 9999 for _ in pathLengthFrom2End]):
            unDiGraph = networkGraph.to_undirected()
            pathFrom2End = []
            pathLengthFrom2End = []
            for i in range(len(fromStopNodes)):
                for j in range(len(endStopNodes)):
                    try:
                        path = nx.shortest_path(unDiGraph,
                                                fromStopNodes[i], endStopNodes[j],
                                                weight='length',
                                                method='dijkstra')
                        pathLength = nx.shortest_path_length(unDiGraph,
                                                             fromStopNodes[i], endStopNodes[j],
                                                             weight='length',
                                                             method='dijkstra')
                    except:
                        path = []
                        # Set a large number to indicate the path is not available
                        pathLength = 9999
                        print('No path found between using unDiGraph: ',
                              fromStopInfo[1][i], 'coords: ', fromStopNodes[2][i],
                              endStopInfo[1][j], 'coords: ', endStopNodes[2][j])
                    pathFrom2End.append(path)
                    pathLengthFrom2End.append(pathLength)

        '''@DeprecationWarning '''
        # pathFrom2End = [nx.shortest_path(networkGraph,
        #                                  fromStopNodes[i], endStopNodes[j],
        #                                  weight='length',
        #                                  method='dijkstra')
        #                 for i in range(len(fromStopNodes))
        #                 for j in range(len(endStopNodes))]
        # pathLengthFrom2End = [nx.shortest_path_length(networkGraph,
        #                                               fromStopNodes[i], endStopNodes[j],
        #                                               weight='length',
        #                                               method='dijkstra')
        #                      for i in range(len(fromStopNodes))
        #                      for j in range(len(endStopNodes))]
        return pathFrom2End, pathLengthFrom2End

    def findLinePath(self,
                     networkGraph: nx.Graph,
                     matchedStopsShp: gpd.GeoDataFrame,
                     preMatchedDict: dict,
                     stopsShpNameCol: str,
                     ):
        '''
        for segId, stopsNameTuple in enumerate(self.consectiveStopsName):
            if segId == 0:

                firstStop = Stop(stopsNameTuple[0],
                                 lon=self.stopsCoordsList[segId*2][0],
                                 lat=self.stopsCoordsList[segId*2][1])
                matchedStopNameList_1st, matchedStopCoordsList_1st = self.deriveStopName(firstStop,
                                                                                         matchedStopsShp,
                                                                                         stopsShpNameCol,
                                                                                         preMatchedDict)
                secondStop = Stop(stopsNameTuple[1],
                                  lon=self.stopsCoordsList[segId*2+1][0],
                                  lat=self.stopsCoordsList[segId*2+1][1])
                matchedStopNameList_2nd, matchedStopCoordsList_2nd = self.deriveStopName(secondStop,
                                                                                         matchedStopsShp,
                                                                                         stopsShpNameCol,
                                                                                         preMatchedDict)
                thirdStop = Stop(self.consectiveStopsName[segId+1][0],
                                 lon=self.stopsCoordsList[(segId+1)*2][0],
                                 lat=self.stopsCoordsList[(segId+1)*2][1])
                matchedStopNameList_3rd, matchedStopCoordsList_3rd = self.deriveStopName(thirdStop,
                                                                                         matchedStopsShp,
                                                                                         stopsShpNameCol,
                                                                                         preMatchedDict)


            elif segId == len(self.consectiveStopsName)-1:
                pass
            else:
                pass'''

        G = nx.MultiDiGraph()
        # Add a fake start node
        G.add_node(0, name='Start',
                   coords=(0, 0)
                   )
        node_id = 1

        avg_seg_length = []

        for segId, stopsNameTuple in enumerate(self.consectiveStopsName):

            if segId == 0:

                fromStop = Stop(stopsNameTuple[0],
                                lon=self.stopsCoordsList[0][0],
                                lat=self.stopsCoordsList[0][1])
                matchedStopNameList_from, matchedStopCoordsList_from = self.deriveStopName(fromStop,
                                                                                           matchedStopsShp,
                                                                                           stopsShpNameCol,
                                                                                           preMatchedDict)
                fromStopIds = [_ + 1 for _ in (range(len(matchedStopNameList_from)))]
                node_id += len(fromStopIds)

                # Add nodes to the graph
                for _idx, _nodeId in enumerate(fromStopIds):
                    G.add_node(_nodeId,
                               name=matchedStopNameList_from[_idx],
                               coords=matchedStopCoordsList_from[_idx])
                    # Add fake edges from node 0 to the real start node(s)
                    G.add_edge(0, _nodeId, length=1)
            else:
                fromStop = endStop
                matchedStopNameList_from = matchedStopNameList_end
                matchedStopCoordsList_from = matchedStopCoordsList_end
                fromStopIds = endStopIds

            ''' End stop of the segment '''
            endStop = Stop(stopsNameTuple[1],
                           lon=self.stopsCoordsList[segId + 1][0],
                           lat=self.stopsCoordsList[segId + 1][1])
            matchedStopNameList_end, matchedStopCoordsList_end = self.deriveStopName(endStop,
                                                                                     matchedStopsShp,
                                                                                     stopsShpNameCol,
                                                                                     preMatchedDict)

            endStopIds = (np.array(range(len(matchedStopNameList_end))) + node_id).tolist()
            node_id += len(endStopIds)

            # Add nodes to the graph
            for _idx, _nodeId in enumerate(endStopIds):
                G.add_node(_nodeId,
                           name=matchedStopNameList_end[_idx],
                           coords=matchedStopCoordsList_end[_idx])

            segPath, segLength = self.findConsectivePath(
                (fromStop, matchedStopNameList_from, matchedStopCoordsList_from),
                (endStop, matchedStopNameList_end, matchedStopCoordsList_end),
                networkGraph)
            # # Add avg_length of this segment
            # avg_seg_length.append(np.mean(segLength))
            '''Add edges to the graph with length as the weight'''
            # print(segLength)
            # print(fromStopIds, endStopIds)
            for s_idx, s_node in enumerate(fromStopIds):
                for e_idx, e_node in enumerate(endStopIds):
                    G.add_edge(s_node, e_node,
                               length=segLength[s_idx * len(endStopIds) + e_idx],
                               path=segPath[s_idx * len(endStopIds) + e_idx])

            ''' Add a fake end node for the last segment'''
            if segId == len(self.consectiveStopsName) - 1:
                # Add a fake end node
                lastStopId = node_id + 1
                G.add_node(lastStopId, name='End', coords=(0, 0))
                for _endNodeId in endStopIds:
                    G.add_edge(_endNodeId, lastStopId, length=1)

        """TODO:
        1. Add and report segment length - Done
        """

        lineShortestPathLength = nx.shortest_path_length(G,
                                                         source=0,
                                                         target=lastStopId,
                                                         weight='length',
                                                         method='dijkstra')
        lineShortestPath = nx.shortest_path(G,
                                            source=0,
                                            target=lastStopId,
                                            weight='length',
                                            method='dijkstra')
        lineSegmentPathLength = [nx.shortest_path_length(G,
                                                         source=lineShortestPath[i],
                                                         target=lineShortestPath[i + 1],
                                                         weight='length',
                                                         method='dijkstra')
                                 for i in range(len(lineShortestPath) - 1)]

        '''
        Check if there are any unusual long segment:
            if yes, get rid of the corresponding stops
        '''

        self.clearStopsCoordsList()
        self.clearStopsNameList()
        self.updateStopsInfo([G.nodes[_name]['name'] for _name in lineShortestPath[1:-1]],
                             [G.nodes[_coords]['coords'] for _coords in lineShortestPath[1:-1]])
        self.set_length(lineShortestPathLength - 2)  # Exclude the fake start and end nodes
        self.set_findPathGraph(G)
        self.set_busPairShortestPath(lineSegmentPathLength)


