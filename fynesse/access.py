from .config import *
import requests
import pymysql
import csv
import time
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import seaborn as sns

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def hello_world():
  print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  print('Data stored for year: ' + str(year))

def buildings_with_addr(latitude, longitude):
  km_to_deg = 1/111
  north, south, east, west = latitude+km_to_deg, latitude-km_to_deg, longitude+km_to_deg, longitude-km_to_deg
  tags = {"building": True}
  buildings = ox.geometries_from_bbox(north, south, east, west, tags)

  addr_cols = ["addr:housenumber", "addr:street", "addr:postcode"]
  buildings_with_addr = buildings.dropna(subset=addr_cols)
  buildings_with_addr['area_sqm'] = buildings.geometry.area * 111000 * 111000
  buildings_with_addr['addr:street'] = buildings_with_addr['addr:street'].str.upper()
  return buildings_with_addr

def plot_area(latitude, longitude, place_name, pois):
  km_to_deg = 1/111
  north, south, east, west = latitude+km_to_deg, latitude-km_to_deg, longitude+km_to_deg, longitude-km_to_deg
  graph = ox.graph_from_bbox(north, south, east, west)
  nodes, edges = ox.graph_to_gdfs(graph)
  area = ox.geocode_to_gdf(place_name)

  fig, ax = plt.subplots()
  area.plot(ax=ax, facecolor="white")
  edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

  ax.set_xlim([west, east])
  ax.set_ylim([south, north])
  ax.set_xlabel("longitude")
  ax.set_ylabel("latitude")

  buildings_with_addr.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
  plt.tight_layout()

def merge(buildings_with_addr, houses_df):
  merged_df = pd.merge(buildings_with_addr, houses_df, left_on=['addr:street'], right_on=['primary_addressable_object_name'], how='inner')
  merged_df[['min_value', 'max_value']] = merged_df['addr:housenumber'].str.split('-', expand=True)

  merged_df['min_value'] = pd.to_numeric(merged_df['min_value'])
  merged_df['max_value'] = pd.to_numeric(merged_df['max_value'])
  merged_df['secondary_addressable_object_name'] = pd.to_numeric(merged_df['secondary_addressable_object_name'])

  merged_df = merged_df[(merged_df['secondary_addressable_object_name'] >= merged_df['min_value']) & (merged_df['secondary_addressable_object_name'] <= merged_df['max_value'])]
  merged_df = merged_df.filter(items=['secondary_addressable_object_name', 'primary_addressable_object_name', 'price', 'date_of_transfer', 'area_sqm'])
  return merged_df

def plot_price_area_corr(merged_df):
  plt.figure()
  sns.scatterplot(x='price', y='area_sqm', data=merged_df)
  plt.title('Price vs Area')
  plt.xlabel('Area (sq m)')
  plt.ylabel('Price')
  plt.show()

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

