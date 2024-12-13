from .config import *

from . import access
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

def string_to_dict(tag_string):
    tag_string = tag_string[1:-1]
    if pd.isna(tag_string):
        return None
    try:
        return {k[1:-1]: v[1:-1] for k, v in (item.split(': ') for item in tag_string.split(', '))}
    except ValueError:
        return None

def plot_elections(region, oa_gdf, engine, party_colors):
  if region == None:
    query = "SELECT * FROM election_data"
  else:
    query = f"SELECT * FROM election_data WHERE Regionname = '{region}'"

  election_df = pd.read_sql_query(query, con=engine)
  oa_gdf = oa_gdf.merge(election_df, left_on="oa21cd", right_on="OA21CD")

  oa_gdf["color"] = oa_gdf["Partyname"].map(party_colors)
  oa_gdf["color"] = oa_gdf["color"].fillna("gray")

  fig, ax = plt.subplots(1, 1, figsize=(12, 12))
  oa_gdf.plot(ax=ax, color=oa_gdf["color"], linewidth=0.1, edgecolor="white")

  ax.set_title("UK 2024 Election Results by Output Area", fontsize=16)
  legend_elements = [plt.Line2D([0], [0], marker="o", color=party_colors[party],
                                label=party, markersize=10) for party in party_colors]
  ax.legend(handles=legend_elements, title="Winning Party", loc="lower left")
  plt.show()

def get_nearby_oas(engine, latitude, longitude, radius_km=1.0):
  km_dg = 1/111
  min_lat, max_lat = latitude-(radius_km*km_dg), latitude+(radius_km*km_dg)
  min_lon, max_lon = longitude-(radius_km*km_dg), longitude+(radius_km*km_dg)

  #%sql USE `ads_2024`;
  #oas = %sql SELECT OA21CD, LAT, LON FROM oa_data WHERE LAT BETWEEN :min_lat AND :max_lat AND LON BETWEEN :min_lon AND :max_lon;
  query = f"SELECT OA21CD, LAT, LON FROM oa_data WHERE LAT BETWEEN {min_lat} AND {max_lat} AND LON BETWEEN {min_lon} AND {max_lon};"
  #oas_df = oas.DataFrame()
  oas_df = pd.read_sql_query(query, con=engine)
  oas_df.columns = ["oa21cd", "lat", "lon"]
  oas_set = set(oas_df["oa21cd"])

  #urban_classificaton = %sql SELECT * FROM rural_urban_data WHERE oa21cd IN :oas_set;
  #urban_classificaton_df = urban_classificaton.DataFrame()
  query2 = f"SELECT * FROM rural_urban_data WHERE oa21cd IN {oas_set};"
  urban_classification_df = pd.read_sql_query(query2, con=engine)
  oas_df = oas_df.merge(urban_classification_df, on="oa21cd", how="inner")

  return oas_df

def get_pois_radius(engine, latitude, longitude, radius_km=1.0):
  km_dg = 1/111
  min_lat, max_lat = latitude-(radius_km*km_dg), latitude+(radius_km*km_dg)
  min_lon, max_lon = longitude-(radius_km*km_dg), longitude+(radius_km*km_dg)

  #pois = %sql SELECT DISTINCT * FROM osm_nodes WHERE lat between :min_lat AND :max_lat AND lon BETWEEN :min_lon AND :max_lon;
  #pois_df = pois.DataFrame()
  query = f"SELECT DISTINCT * FROM osm_nodes WHERE lat between {min_lat} AND {max_lat} AND lon BETWEEN {min_lon} AND {max_lon};"
  pois_df = pd.read_sql_query(query, con=engine)
  return pois_df

def nearest_oa(pois_df, oa_df):
  pois_df["tags"] = pois_df["tags"].apply(string_to_dict)
  oa_gdf_points = gpd.GeoDataFrame(oa_df, geometry=gpd.points_from_xy(oa_df['lon'], oa_df['lat']), crs="EPSG:3857")
  pois_gdf = gpd.GeoDataFrame(pois_df, geometry=gpd.points_from_xy(pois_df['lon'], pois_df['lat']),crs="EPSG:3857")

  oa_coords = np.array(list(zip(oa_gdf_points.geometry.x, oa_gdf_points.geometry.y)))
  pois_coords = np.array(list(zip(pois_gdf.geometry.x, pois_gdf.geometry.y)))

  tree = cKDTree(oa_coords)
  distances, indices = tree.query(pois_coords)
  pois_gdf['nearest_oa'] = oa_gdf_points.iloc[indices]['oa21cd'].values
  pois_df_new = pois_gdf.drop(columns=['geometry'])

  return pois_df_new

def pois_by_oa(oas, pois_df):
  oa_gdf_points = gpd.GeoDataFrame(oas, geometry=gpd.points_from_xy(oas['lon'], oas['lat']), crs="EPSG:3857")
  pois_gdf = gpd.GeoDataFrame(pois_df, geometry=gpd.points_from_xy(pois_df['lon'], pois_df['lat']),crs="EPSG:3857")

  oa_coords = np.array(list(zip(oa_gdf_points.geometry.x, oa_gdf_points.geometry.y)))
  pois_coords = np.array(list(zip(pois_gdf.geometry.x, pois_gdf.geometry.y)))

  tree = cKDTree(oa_coords)
  distances, indices = tree.query(pois_coords)
  pois_gdf['nearest_oa'] = oa_gdf_points.iloc[indices]['oa21cd'].values
  pois_df_new = pois_gdf.drop(columns=['geometry'])

  for key in keys:
    pois_df_new[key] = pois_df_new["tags"].apply(lambda tags: 1 if tags and key in tags else 0)

  osm_totals = pois_df_new.groupby("nearest_oa")[keys].sum().reset_index()
  osm_totals["total"] = osm_totals[keys].sum(axis=1)
  oas_with_counts = oas.merge(osm_totals, left_on="oa21cd" ,right_on="nearest_oa", how="left")
  oas_with_counts[keys + ["total"]] = oas_with_counts[keys + ["total"]].fillna(0)

  columns_to_normalize = keys + ["total"]
  oas_with_counts[columns_to_normalize] = oas_with_counts[columns_to_normalize].applymap(lambda x: np.log1p(x))
  oas_with_counts[columns_to_normalize] = scaler.fit_transform(oas_with_counts[columns_to_normalize])
  oas_with_counts[columns_to_normalize] = oas_with_counts[columns_to_normalize].fillna(0)
  oas_with_counts = oas_with_counts[["oa21cd"] + columns_to_normalize]
  return oas_with_counts

def get_postcodes_from_oas(engine, oas):
  oas_set = set(oas['oa21cd'])
  #%sql USE `ads_2024`;
  #postcodes = %sql SELECT oa21cd, pcd7 FROM postcode_oa_lookup WHERE oa21cd in :oas_set;
  #postcodes_df = postcodes.DataFrame()
  query = f"SELECT oa21cd, pcd7 FROM postcode_oa_lookup WHERE oa21cd in {oas_set};"
  postcodes_df = pd.read_sql_query(query, con=engine)
  postcodes_df.columns = ["oa21cd", "postcode"]
  return postcodes_df

def get_house_transactions_from_oas(engine, oas):
  postcodes = get_postcodes_from_oas(oas)
  postcodes_set = set(postcodes["postcode"])
  #%sql USE `ads_2024`;
  #houses = %sql SELECT price, postcode, property_type FROM prices_coordinates_data WHERE postcode in :postcodes_set;
  #houses_df = houses.DataFrame()
  query = f"SELECT price, postcode, property_type FROM prices_coordinates_data WHERE postcode in {postcodes_set};"
  houses_df = pd.read_sql_query(query, con=engine)
  houses_df = houses_df.merge(postcodes, on="postcode")

  houses_price = houses_df.groupby("oa21cd")["price"].mean().reset_index()
  houses_price["avg_price"] = houses_price["price"].apply(lambda x: (int(x)//100)*1000)
  houses_price["log_avg_price"] = houses_price["price"].apply(lambda x: np.log1p(x))
  houses_price = houses_price.drop(columns=["price"])

  houses_price["norm_price"] = scaler.fit_transform(houses_price[["avg_price"]])
  houses_price["log_norm_price"] = scaler.fit_transform(houses_price[["log_avg_price"]])
  oas_with_prices = oas.merge(houses_price, on="oa21cd", how="left")

  columns_to_fill = ["avg_price", "log_avg_price", "norm_price", "log_norm_price"]
  oas_with_prices[columns_to_fill] = oas_with_prices[columns_to_fill].fillna(0)
  oas_with_prices = oas_with_prices.drop(columns=["lat", "lon", "urban"])
  return oas_with_prices

def election_from_oas(engine, oas):
  oas_set = set(oas["oa21cd"])
  #%sql USE `ads_2024`;
  #election_oa = %sql SELECT OA21CD, Partyname, Share FROM election_data WHERE OA21CD in :oas_set;
  #election_oa_df = election_oa.DataFrame()
  query = f"SELECT OA21CD, Partyname, Share FROM election_data WHERE OA21CD in {oas_set};"
  election_oa_df = pd.read_sql_query(query, con=engine)
  election_oa_df.columns = ["oa21cd", "party", "share"]
  return election_oa_df

def census_from_oas(engine, oas):
  oas_set = set(oas["oa21cd"])
  #%sql USE `ads_2024`;
  #l4 = %sql SELECT OA21CD, L4, total FROM qualification_data WHERE OA21CD IN :oas_set;
  #l4_df = l4.DataFrame()
  query = f"SELECT OA21CD, L4, total FROM qualification_data WHERE OA21CD IN {oas_set};"
  l4_df = pd.read_sql_query(query, con=engine)
  l4_df.columns = ["oa21cd", "L4", "population"]
  l4_df["L4_percent"] = l4_df["L4"]/l4_df["population"]
  l4_df["L4_norm_percent"] = scaler.fit_transform(l4_df[["L4_percent"]])

  return l4_df

def combined_dfs_on_oa(oas, dfs):
  final_df = oas.copy()
  for df in dfs:
    final_df = final_df.merge(df, on="oa21cd")
  return final_df

def alpha(val):
  if val == 0:
    return 0.1
  else:
    return val

def plot_oas_election(df, oa_gdf, fig_ax=None, pos=None):
  share_col = df["share"]
  df["norm_share"] = (share_col - share_col.min()) / (share_col.max() - share_col.min())
  oa_gdf = oa_gdf.merge(df, on="oa21cd")

  oa_gdf["color"] = oa_gdf["party"].map(party_colors)
  oa_gdf["color"] = oa_gdf["color"].fillna("gray")
  alphas = oa_gdf["share"].apply(alpha)

  if not fig_ax:
    fig, ax = plt.subplots(1, 1)
  else:
    fig, ax = fig_ax
  
  if pos:
    ax = ax[pos[0], pos[1]]
  oa_gdf.plot(ax=ax, color=oa_gdf["color"], linewidth=0.5, edgecolor="black", alpha=alphas)

  ax.set_title("Election Results by Select Output Area", fontsize=8)
  ax.set_xticks([])
  ax.set_yticks([])

def plot_oas_osm(df, oa_gdf, fig_ax=None, pos=None):
  oa_gdf = oa_gdf.merge(df, on="oa21cd")
  alphas = oa_gdf["total"].apply(alpha)

  if not fig_ax:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  else:
    fig, ax = fig_ax
  
  if pos:
    ax = ax[pos[0], pos[1]]
  oa_gdf.plot(ax=ax, color="gray", linewidth=0.5, edgecolor="black", alpha=alphas)

  ax.set_title("Amount of OSM nodes by Select Output Area", fontsize=8)
  ax.set_xticks([])
  ax.set_yticks([])

def plot_oas_houses_price(df, oa_gdf, fig_ax=None, pos=None):
  oa_gdf = oa_gdf.merge(df, on="oa21cd")
  alphas = oa_gdf["log_norm_price"].apply(alpha)

  if not fig_ax:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  else:
    fig, ax = fig_ax
  
  if pos:
    ax = ax[pos[0], pos[1]]
  oa_gdf.plot(ax=ax, color="gray", linewidth=0.5, edgecolor="black", alpha=alphas)

  ax.set_title("Average log house price by Select Output Area", fontsize=8)
  ax.set_xticks([])
  ax.set_yticks([])

def plot_oas_degree_qualified(df, oa_gdf, fig_ax=None, pos=None):
  oa_gdf = oa_gdf.merge(df, on="oa21cd")
  alphas = oa_gdf["L4_norm_percent"].apply(alpha)

  if not fig_ax:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  else:
    fig, ax = fig_ax

  if pos:
    ax = ax[pos[0], pos[1]]
  oa_gdf.plot(ax=ax, color="gray", linewidth=0.5, edgecolor="black", alpha=alphas)

  ax.set_title("Percent with degree level qualification Select Output Area", fontsize=8)
  ax.set_xticks([])
  ax.set_yticks([])

def plot_oas_rural_urban(df, oa_gdf, fig_ax=None, pos=None):
  oa_gdf = oa_gdf.merge(df, on="oa21cd")
  alphas = oa_gdf["urban"].apply(alpha)

  if not fig_ax:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  else:
    fig, ax = fig_ax

  if pos:
    ax = ax[pos[0], pos[1]]
  oa_gdf.plot(ax=ax, color="gray", linewidth=0.5, edgecolor="black", alpha=alphas)

  ax.set_title("Rural or Urban by Select Output Area", fontsize=8)
  ax.set_xticks([])
  ax.set_yticks([])


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
