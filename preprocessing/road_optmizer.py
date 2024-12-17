"""
Author: Xander Peng
Date: 2024/12/17
Description: Run this script to process the raw network into bidirectional road network.
    Note that the output road network should be further processed (preferably using gis software)
    combined with the "projected stops" (@link: raw_stop2network_projection.py) to get the "splitDiGovRoad.shp".
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString


def process_directional_network(network: gpd.GeoDataFrame,
                                output_dir: str,
                                ):

    # Create a reversed way DF for road segments with "TRAVEL_DIR" == 1
    reversedRoads = network[network['TRAVEL_DIR'] == 1].copy()
    reversedRoads['geometry'] = reversedRoads['geometry'].map(lambda x: LineString(x.coords[::-1]))
    reversedRoads['ROUTE_ID'] = reversedRoads['ROUTE_ID'].astype(str)
    reversedRoads['ROUTE_ID'] = reversedRoads['ROUTE_ID'].map(lambda x: x + '_r')

    # Concatenate the original and reversed road segments
    diGovRoadWGS84 = pd.concat([network, reversedRoads])

    # some magic to reset the index
    diGovRoadWGS84.sort_index(inplace=True)
    diGovRoadWGS84.reset_index(drop=False, inplace=True)
    diGovRoadWGS84.rename(columns={'index': 'origidx'}, inplace=True)

    if output_dir is not None:
        diGovRoadWGS84.to_file(output_dir+'/fullDiGovRoad.shp')

if __name__ == '__main__':

    ''' Read the raw road network data '''
    govRoad = gpd.read_file('../../Data/text-file/gov-data/road/govDiRoad.shp')
    ''' Convert the road network to EPSG:4326 if it is not '''
    if govRoad.crs is None or govRoad.crs != 'EPSG:4326':
        govRoad = govRoad.to_crs('EPSG:4326')

    ''' Process the road network '''
    process_directional_network(network=govRoad,
                                output_dir=r'../data/interim/road_network/')