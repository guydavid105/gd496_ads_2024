# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""
def votes_by_urban(urban, voting, parties):
  voting_by_cat = voting[voting["urban"] == urban]

  average_votes_share = voting_by_cat.groupby("party")["share"].mean().reset_index()
  average_votes_pop = voting_by_cat.groupby("party")["population"].sum().reset_index()
  average_votes = average_votes_share.merge(average_votes_pop, on="party")

  average_votes["norm_population"] = average_votes["population"] / sum(average_votes["population"])
  average_votes["adjusted_share"] = average_votes["share"] * average_votes["norm_population"]

  for party in parties:
    if party not in set(average_votes["party"]):
      average_votes.loc[len(average_votes)] = [party, 0, 0, 0, 0]
  average_votes = average_votes.sort_values(by="party", ascending=True)

  return average_votes

def urban_rural_diff(votes_urban, votes_rural):
  diff = abs(votes_urban["adjusted_share"] - votes_rural["adjusted_share"])
  diff = sum(diff)
  return diff

def urban_rural_diffs(df, keys_to_train, parties):
  voting = df[["oa21cd", "urban", "party", "share", "population"]]
  urban = df[df["urban"] == 1]
  rural = df[df["urban"] == 0]

  urban = urban[keys_to_train]
  rural = rural[keys_to_train]
  avgs_urban = {}
  avgs_rural = {}

  for col in keys_to_train:
    avgs_urban[col] = [sum(urban[col])/len(urban[col])]
    avgs_rural[col] = [sum(rural[col])/len(rural[col])]

  urban = pd.DataFrame(avgs_urban)
  rural = pd.DataFrame(avgs_rural)

  voting_adjusted_urban = votes_by_urban(True, voting, parties)
  voting_adjusted_rural = votes_by_urban(False, voting, parties)
  voting_diff = urban_rural_diff(voting_adjusted_urban, voting_adjusted_rural)

  diffs = urban - rural
  diffs.columns = [key+"_diff" for key in keys_to_train]
  diffs["voting_diff"] = [voting_diff]
  return diffs
