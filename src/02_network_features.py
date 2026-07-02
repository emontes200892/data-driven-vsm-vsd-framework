import pandas as pd
import networkx as nx
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT/'data', ROOT/'results'

df = pd.read_csv(DATA/'synthetic_event_log.csv', parse_dates=['start_ts','end_ts']).sort_values(['case_id','start_ts'])
# Activity directly-follows graph
G = nx.DiGraph()
for _, g in df.groupby('case_id'):
    acts = g.activity.tolist()
    for a,b in zip(acts[:-1], acts[1:]):
        if G.has_edge(a,b): G[a][b]['weight'] += 1
        else: G.add_edge(a,b,weight=1)

# Centralities
undirected = G.to_undirected()
features = pd.DataFrame({'activity': list(G.nodes())})
features['degree_centrality'] = features.activity.map(nx.degree_centrality(undirected))
features['betweenness_centrality'] = features.activity.map(nx.betweenness_centrality(undirected, weight=None))
features['closeness_centrality'] = features.activity.map(nx.closeness_centrality(undirected))
try:
    eig = nx.eigenvector_centrality(undirected, max_iter=1000)
except Exception:
    eig = {n: 0 for n in G.nodes()}
features['eigenvector_centrality'] = features.activity.map(eig)
features.to_csv(RESULTS/'network_activity_features.csv', index=False)

edges = pd.DataFrame([(u,v,d['weight']) for u,v,d in G.edges(data=True)], columns=['source','target','frequency'])
edges.to_csv(RESULTS/'dfg_edges.csv', index=False)
print(features.round(4))
