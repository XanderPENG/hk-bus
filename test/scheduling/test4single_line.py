"""
Author: Xander Peng
Date: 2024/10/22
Description: 
"""

import pandas as pd
from scheduling.bus_line_eta import BusLineEta
from scheduling.eta import Eta
import scheduling.tools as tools
import logging

"""
Here, I added the field name of each column in the data file manually;
set a default nameList after reading the data file could be a good option if too many files are involved.
"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test4single_line')

logger.info("Reading the test data...")
test_data_dir = r'../../data/input/test/oneLineOneDayETA.csv'
test_data = pd.read_csv(test_data_dir)
test_data['eta'] = pd.to_datetime(test_data['eta'])
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
logger.info("Data reading completed.")

# Create BusRoute instance
etas = [Eta(idx=idx, co=row.co, line_id=row.route, direction=row.dir, service_type=row.service_type, seq=row.seq,
            dest=row.dest,eta_seq=row.eta_seq, eta=row.eta, rmk=row.rmk, scrapped_time=row.timestamp)
            for idx, row in test_data.iterrows()]
test_line = BusLineEta(co=etas[0].co ,etas=etas, line_id=etas[0].line_id,
                        dir=etas[0].dir, service_type=etas[0].service_type,
                        dest=etas[0].dest, rmk=etas[0].rmk)

# Backward search routes
logger.info("Backward searching feasible routes...")
test_line.backward_search_feasible_routes()
time_interval_dict = test_line.all_time_intervals
logger.info("Backward searching completed, with {} candidate routes.".format(len(test_line.routes)))

complete_routes = list(filter(lambda route: route.is_complete() is True , test_line.routes))
incomplete_routes = list(filter(lambda route: route.is_complete() is False , test_line.routes)) # or filter out the complete routes for the sake of efficiency

logger.warning(f"Complete routes (initial): {len(complete_routes)}")
logger.warning(f"Incomplete routes (initial): {len(incomplete_routes)}")

# Drop the duplicates of complete routes
logger.info("Dropping duplicates of complete routes...")
complete_routes, merged_mapping = tools.find_and_merge_duplicates(complete_routes)
logger.warning(f"Complete routes (after dropping duplicates): {len(complete_routes)}")

# Merge the incomplete routes
logger.info("Merging incomplete routes...")
merged_routes = tools.merge_all_routes(incomplete_routes, time_interval_dict)
logger.warning(f"merged routes: {len(merged_routes)}")

# Filter out the complete routes from the merged routes
complete_routes_after_merging = list(filter(lambda route: route.is_complete() is True , merged_routes))
incomplete_routes_after_merging = list(filter(lambda route: route.is_complete() is False , merged_routes))
logger.warning(f"Complete routes (after merging): {len(complete_routes_after_merging)}")
logger.warning(f"Incomplete routes (after merging): {len(incomplete_routes_after_merging)}")

''' Here, we can use the average time interval to "fill" the missing etas in the incomplete routes. 
    However, it is not my recommendation to do so, instead, we should drop them as they are not reliable.
    The example of filling the missing etas as below:
'''

# for route in incomplete_routes_after_merging:
#     route.complete_route_by_avg_etas(time_interval_dict)

# Output the complete routes
all_complete_routes = complete_routes + complete_routes_after_merging
logger.warning(f"Output complete routes with the number of: {len(all_complete_routes)}")
tools.output_complete_routes(all_complete_routes, r'../../data/interim/scheduling/test_complete_routes.csv')


