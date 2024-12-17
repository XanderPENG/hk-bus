"""
Author: Xander Peng
Date: 2024/12/17
Description: Run this script to project the raw stops to the network.
"""

from bus_line import Stop
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging


def proj_stop2Network(rawLineStops: gpd.GeoDataFrame,
                      roadNetwork: gpd.GeoDataFrame,
                      **kwargs
                      ):
    # A list of all bus stop instances
    stations = []
    # Create bus stop object and append to the list
    for idx, row in rawLineStops.iterrows():
        station = Stop(row[kwargs.get('station_name_col')], row[kwargs.get('lon_col')], row[kwargs.get('lat_col')])
        stations.append(station)
    # Find the nearest proj-point of bus stop on the road network
    error_stations = []
    proj_stops = []  # A list of all projected points with DataFrame format
    count = 0
    for _stop in stations:
        try:
            _proj_stops = _stop.getProjectionPointsinNearestLines(30, roadNetwork)  # 30 meters
            proj_stops.append(_proj_stops)
        except:
            error_stations.append(_stop)
            logging.warning(f'Error: {_stop.stopName}')
        count += 1
        logging.info(f'Count: {count}')

    aggProjStops = pd.concat(proj_stops)
    aggProjStops['lon'] = aggProjStops['coords'].str[0]
    aggProjStops['lat'] = aggProjStops['coords'].str[1]
    aggProjStops['geometry'] = [Point(lon, lat) for lon, lat in zip(aggProjStops['lon'], aggProjStops['lat'])]
    aggProjStops = gpd.GeoDataFrame(aggProjStops, geometry='geometry', crs='EPSG:4326')
    aggProjStops.reset_index(drop=False, inplace=True)

    ''' Filter out the stops that are totally the same (including name, coords)'''
    filteredProjStops = aggProjStops.drop_duplicates(subset=['name', 'coords'])
    filteredProjStops.rename(columns={'index': 'seqStopidx'}, inplace=True)
    filteredProjStops.drop(columns='coords', inplace=True)

    return filteredProjStops

if __name__ == '__main__':

    ''' Read the raw stop-line data '''
    raw_line_stops = gpd.read_file('../data/input/rawLineStops/rawLineStops.shp')

    ''' Read the raw road network data '''
    govRoad = gpd.read_file('../../Data/text-file/gov-data/road/govDiRoad.shp')

    ''' Convert the road network to EPSG:4326 if it is not '''
    if govRoad.crs is None or govRoad.crs != 'EPSG:4326':
        govRoad = govRoad.to_crs('EPSG:4326')

    ''' Project the stops to the network '''
    filtered_proj_stops = proj_stop2Network(rawLineStops=raw_line_stops,
                                            roadNetwork=govRoad,
                                            ## The following are the column names of the raw stop data
                                            station_name_col='stationLis',
                                            lon_col='Lon',
                                            lat_col='Lat')

    ''' Save the projected stops to a shapefile '''


