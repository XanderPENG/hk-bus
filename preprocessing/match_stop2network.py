"""
Author: Xander Peng
Date: 2024/8/19
Description:
    This script is used to match the stops to the network.
    Replace the key input @raw_line_stops if you

"""

import geopandas as gpd
import networkx as nx
from bus_line import Stop, Line
import utils

''' Read the stops and network data '''
raw_line_stops = gpd.read_file('../data/input/rawLineStops/rawLineStops.shp')  # raw shp-format line-stop info
gov_road = gpd.read_file('../data/input/road/splitDiGovRoadV2.shp')  # Use V2 for better accuracy
proj_stops = gpd.read_file('../data/input/proj_stops/filteredAggProjStops.shp')  # projected stops

road_graph, indexed_road = utils.create_graph(gov_road)

if __name__ == '__main__':
    # Group the line-stop info by line_name, co, and direction, so that we can match the stops to the network by line
    grouped_line_info = raw_line_stops.groupby(['line_name', 'co', 'direction'])
    grouped_line_info = {name: group for name, group in grouped_line_info}

    # Create a list of Line instances
    line_dfs = [grouped_line_info[line_key] for line_key in list(grouped_line_info.keys())]
    lines = []
    for idx, line_df in enumerate(line_dfs):
        name_list = line_df['stationLis'].tolist()  # List of station names; the field name might be different
        coords_list = list(zip(line_df['Lon'], line_df['Lat']))
        _line = Line(lineId=idx,  # No lineId in the raw data
                     lineName=line_df['line_name'].iloc[0],
                     lineDirection=line_df['direction'].iloc[0],
                     stopsNameList=name_list,
                     stopsCoordsList=coords_list)
        lines.append(_line)

    # Match the stops to the network
    errorLines = []
    processedLines = []
    processedNum = 0
    for line in lines:
        try:
            line.findLinePath(networkGraph=road_graph,
                              matchedStopsShp=proj_stops,
                              preMatchedDict={},  # No preMatchedDict
                              stopsShpNameCol='name')
            processedLines.append(line)
        except nx.NetworkXException:
            print(
                f'Error: (ID: {line.lineId}); (Name: {line.get_name()}); (Direction: {line.get_direction()})]')
            errorLines.append(line)
        processedNum += 1
        print(f'Processed: {processedNum}')

    # Format the matched stops
    result = utils.format_results(processedLines,
                                  output_dir='../data/output/matched_stops//')

    ''' 
    Example: output one of the matched bus line route in shp-format, 
    Use a for-loop to get all the line routes if needed 
    '''
    _path, _pathCoords, _pathLines = utils.getRoutePathLine(road_graph,
                                                            processedLines[2].get_stopsCoordsList(),
                                                            indexed_road)

    gpd.GeoDataFrame(geometry=_pathLines).to_file('../data/output/line_routes/testRoute.shp',
                                                  encoding='utf-8')
