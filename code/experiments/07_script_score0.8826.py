# ============================================================
# IMPORTS
# ============================================================

import csv
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ============================================================
# 1️⃣ Parse CSV
# ============================================================

def parse_adjacency_list(file_path):
    edges = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            source = str(row[0]).strip()

            for target in row[1:]:
                target = str(target).strip()
                if target:
                    edges.append((source, target))

    return edges


# ============================================================
# 2️⃣ Easy Negative Sampling ONLY
# ============================================================

def sample_easy_negatives(G, n):
    nodes = list(G.nodes())
    negatives = set()

    while len(negatives) < n:
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            negatives.add((u, v))

    return list(negatives)


# ============================================================
# 3️⃣ Precompute
# ============================================================

def precompute(G):

    in_cache = {}
    out_cache = {}
    degree_cache = {}

    for node in G.nodes():
        u_in = set(G.predecessors(node))
        u_out = set(G.successors(node))

        in_cache[node] = u_in
        out_cache[node] = u_out
        degree_cache[node] = len(u_in | u_out)

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    return in_cache, out_cache, degree_cache, in_degrees, out_degrees


# ============================================================
# 4️⃣ FULL 27 FEATURE EXTRACTION (UNCHANGED)
# ============================================================

def extract_features(u, v, G,
                     in_cache, out_cache,
                     degree_cache,
                     in_degrees, out_degrees):

    u_in = in_cache.get(u, set())
    u_out = out_cache.get(u, set())
    v_in = in_cache.get(v, set())
    v_out = out_cache.get(v, set())

    u_all = u_in | u_out
    v_all = v_in | v_out

    u_d = degree_cache.get(u, 0)
    v_d = degree_cache.get(v, 0)

    u_d_in = len(u_in)
    u_d_out = len(u_out)
    u_d_bi = len(u_in & u_out)

    v_d_in = len(v_in)
    v_d_out = len(v_out)
    v_d_bi = len(v_in & v_out)

    common = u_all & v_all
    union = u_all | v_all

    cn_count = len(common)
    jaccard = cn_count / len(union) if union else 0

    transitive_uv = len(u_out & v_in)
    transitive_vu = len(v_out & u_in)
    total_2hop = transitive_uv + transitive_vu

    pref_attach = u_d * v_d

    adamic = 0
    resource = 0

    for z in common:
        deg_z = degree_cache.get(z, 0)
        if deg_z > 1:
            adamic += 1 / np.log(deg_z)
        if deg_z > 0:
            resource += 1 / deg_z

    return [
        u_d, u_d_in, u_d_out, u_d_bi,
        v_d, v_d_in, v_d_out, v_d_bi,
        u_d_in / u_d if u_d else 0,
        u_d_out / u_d if u_d else 0,
        v_d_in / v_d if v_d else 0,
        v_d_out / v_d if v_d else 0,
        cn_count,
        len(u_in & v_in),
        len(u_out & v_out),
        jaccard,
        transitive_uv,
        transitive_vu,
        total_2hop,
        pref_attach,
        abs(u_d - v_d),
        abs(u_d_in - v_d_in),
        abs(u_d_out - v_d_out),
        adamic,
        resource,
        in_degrees.get(v, 0),
        out_degrees.get(u, 0)
    ]


# ============================================================
# 5️⃣ Build Training Dataset (40k Pos / 150k Neg)
# ============================================================

def build_training_dataset(train_edges_full):

    print("Sampling 80,000 positives...")
    positives = random.sample(train_edges_full, 80000)

    remaining_edges = list(set(train_edges_full) - set(positives))

    all_nodes = set()
    for u, v in train_edges_full:
        all_nodes.add(u)
        all_nodes.add(v)

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(remaining_edges)

    print("Sampling 250,000 easy negatives...")
    negatives = sample_easy_negatives(G, 250000)

    in_cache, out_cache, degree_cache, in_deg, out_deg = precompute(G)

    X = []
    y = []

    print("Building dataset...")

    for u, v in positives:
        X.append(extract_features(u, v, G,
                                  in_cache, out_cache,
                                  degree_cache, in_deg, out_deg))
        y.append(1)

    for u, v in negatives:
        X.append(extract_features(u, v, G,
                                  in_cache, out_cache,
                                  degree_cache, in_deg, out_deg))
        y.append(0)

    return np.array(X), np.array(y)


# ============================================================
# 6️⃣ LOAD DATA
# ============================================================

edges = parse_adjacency_list("/kaggle/input/datasets/yakshithm/project-1/train.csv")
train_edges, _ = train_test_split(edges, test_size=0.2, random_state=42)

X, y = build_training_dataset(train_edges)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 7️⃣ Hyperparameter Tuning
# ============================================================

models = {
    "RandomForest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [300, 500, 700],
            "max_depth": [None, 20, 30],
            "min_samples_leaf": [1, 3, 5]
        }
    ),
    "XGBoost": (
        XGBClassifier(
            tree_method="hist",
            device="cuda",
            eval_metric="logloss",
            random_state=42
        ),
        {
            "max_depth": [6, 8, 10],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [400, 600, 800],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9]
        }
    )
}

results = {}

for name, (model, param_grid) in models.items():

    print(f"\nTuning {name}...")

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="roc_auc",
        cv=skf,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)

    best_model = search.best_estimator_
    best_score = search.best_score_

    print(f"{name} Best CV AUC: {best_score:.4f}")
    print("Best Params:", search.best_params_)

    results[name] = (best_score, best_model)


# ============================================================
# 8️⃣ Select Best Model
# ============================================================

best_model_name = max(results, key=lambda x: results[x][0])
best_model = results[best_model_name][1]

print("\nBest Model:", best_model_name)

best_model.fit(X, y)


# ============================================================
# 9️⃣ Build Full Graph for Test Prediction
# ============================================================

all_nodes = set()
for u, v in edges:
    all_nodes.add(u)
    all_nodes.add(v)

G_full = nx.DiGraph()
G_full.add_nodes_from(all_nodes)
G_full.add_edges_from(edges)

in_cache, out_cache, degree_cache, in_deg, out_deg = precompute(G_full)

test_df = pd.read_csv("/kaggle/input/datasets/yakshithm/project-1/test.csv")

X_test = []

print("Extracting test features...")

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    u = str(row["From"]).strip()
    v = str(row["To"]).strip()

    X_test.append(
        extract_features(u, v,
                         G_full,
                         in_cache, out_cache,
                         degree_cache,
                         in_deg, out_deg)
    )

X_test = np.array(X_test)

print("Predicting...")

probs = best_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "Predictions": probs
})

submission.to_csv("submission.csv", index=False)

print("Submission saved.")