import unittest

import geopandas as gpd
import networkx as nx
import numpy as np

from preprocessing.bus_line import Stop, Line
import preprocessing.utils as utils


class MyTestCase(unittest.TestCase):
    thold = 10
    def setUp(self):
        ''' Read the stops and network data '''
        self.raw_line_stops = gpd.read_file('../data/input/rawLineStops/rawLineStops.shp')  # raw shp-format line-stop info
        # Only use 3 lines for testing
        self.raw_line_stops = self.raw_line_stops.query("line_name in ['273A Line', '299X Line', 'A36 Line']")
        self.gov_road = gpd.read_file('../data/input/road/splitDiGovRoad.shp')
        self.proj_stops = gpd.read_file('../data/input/proj_stops/filteredAggProjStops.shp')  # projected stops

        self.road_graph, self.indexed_road = utils.create_graph(self.gov_road)

        # Group the line-stop info by line_name, co, and direction
        grouped_line_info = self.raw_line_stops.groupby(['line_name', 'co', 'direction'])
        self.grouped_line_info = {name: group for name, group in grouped_line_info}

        # Create a list of Line instances
        line_dfs = [self.grouped_line_info[line_key] for line_key in list(self.grouped_line_info.keys())]
        self.lines = []
        for idx, line_df in enumerate(line_dfs):
            name_list = line_df['stationLis'].tolist()  # List of station names; the field name might be different
            coords_list = list(zip(line_df['Lon'], line_df['Lat']))
            _line = Line(lineId=idx,  # No lineId in the raw data
                         lineName=line_df['line_name'].iloc[0],
                         lineDirection=line_df['direction'].iloc[0],
                         stopsNameList=name_list,
                         stopsCoordsList=coords_list)
            self.lines.append(_line)

        # Match the stops to the network
        self.errorLines = []
        self.processedLines = []
        self.processedNum = 0
        for line in self.lines:
            try:
                line.findLinePath(networkGraph=self.road_graph,
                                  matchedStopsShp=self.proj_stops,
                                  preMatchedDict={},  # No preMatchedDict
                                  stopsShpNameCol='name')
                self.processedLines.append(line)
            except nx.NetworkXException:
                print(
                    f'Error: (ID: {line.lineId}); (Name: {line.get_name()}); (Direction: {line.get_direction()})]')
                self.errorLines.append(line)
            self.processedNum += 1
            print(f'Processed: {self.processedNum}')

        # Format the matched stops
        result = utils.format_results(self.processedLines)

        self.busPairShortestPaths = np.array(result['lineBusPairShortestPath'])


    def test_something(self):
        self.assertTrue(np.all(self.busPairShortestPaths<self.thold), "Wrong bus pair length") # add assertion here


if __name__ == '__main__':
    unittest.main()
