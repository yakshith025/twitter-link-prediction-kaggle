# ============================================================
# IMPORTS
# ============================================================

import csv
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ============================================================
#  Parse CSV
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
#  Easy Negative Sampling
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
#  Precompute
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
#  Feature Extraction (27 Features)
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
#  Build Training Dataset
# ============================================================

def build_training_dataset(train_edges_full):

    print("Sampling 40k positives")

    positives = random.sample(train_edges_full, 40000)

    remaining_edges = list(set(train_edges_full) - set(positives))

    all_nodes = set()

    for u, v in train_edges_full:
        all_nodes.add(u)
        all_nodes.add(v)

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(remaining_edges)

    print("Sampling 150k negatives")

    negatives = sample_easy_negatives(G, 150000)

    in_cache, out_cache, degree_cache, in_deg, out_deg = precompute(G)

    X = []
    y = []

    print("Building dataset")

    for u, v in positives:

        X.append(extract_features(u, v, G,
                                  in_cache, out_cache,
                                  degree_cache,
                                  in_deg, out_deg))
        y.append(1)

    for u, v in negatives:

        X.append(extract_features(u, v, G,
                                  in_cache, out_cache,
                                  degree_cache,
                                  in_deg, out_deg))
        y.append(0)

    return np.array(X), np.array(y)



# ============================================================
#  Load Data
# ============================================================

edges = parse_adjacency_list(
    "../data/train.csv"
)

# Build dataset
X, y = build_training_dataset(edges)


# ============================================================
#  Train / Test Split
# ============================================================

X_train, X_test_holdout, y_train, y_test_holdout = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ============================================================
#  Models
# ============================================================

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        random_state=42
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )
}


# ============================================================
#  Stratified K Fold Cross Validation (TRAIN ONLY)
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

for name, model in models.items():

    print("\nTraining", name)

    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

        print("Fold:", fold + 1)

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_val)[:,1]

        auc = roc_auc_score(y_val, probs)

        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)

    print("Mean CV AUC:", mean_auc)

    results[name] = (mean_auc, model)


# ============================================================
#  Select Best Model
# ============================================================

best_model_name = max(results, key=lambda x: results[x][0])

best_auc, best_model = results[best_model_name]

print("\nBest Model:", best_model_name)
print("Best CV AUC:", best_auc)


# ============================================================
#  Train Best Model on FULL TRAIN DATA
# ============================================================

best_model.fit(X_train, y_train)


# ============================================================
# Evaluate on HOLD-OUT TEST SET
# ============================================================

test_probs = best_model.predict_proba(X_test_holdout)[:,1]
test_preds = best_model.predict(X_test_holdout)

test_auc = roc_auc_score(y_test_holdout, test_probs)

print("\nTest AUC:", test_auc)


# ============================================================
#  ROC Curve (TEST DATA)
# ============================================================

fpr, tpr, _ = roc_curve(y_test_holdout, test_probs)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {test_auc:.2f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve (Test Data)")
plt.legend()

plt.show()


# ============================================================
#  Confusion Matrix (TEST DATA)
# ============================================================

cm = confusion_matrix(y_test_holdout, test_preds)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Data)")

plt.show()

# ============================================================
# Feature Importance (All 27 Features)
# ============================================================

feature_names = [

"u_d","u_d_in","u_d_out","u_d_bi",
"v_d","v_d_in","v_d_out","v_d_bi",
"u_in_ratio","u_out_ratio",
"v_in_ratio","v_out_ratio",
"cn_count","common_in","common_out",
"jaccard",
"transitive_uv","transitive_vu","total_2hop",
"pref_attach",
"deg_diff","in_deg_diff","out_deg_diff",
"adamic","resource",
"in_degree_v","out_degree_u"
]

# Extract importance
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
else:
    importance = np.abs(best_model.coef_[0])


# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# Print importance values
print("\nFeature Importance (Sorted):\n")
print(importance_df)


# ============================================================
# Plot Feature Importance
# ============================================================

plt.figure(figsize=(12,6))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xticks(rotation=90)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.title("Feature Importance (27 Features)")

plt.show()

# ============================================================
#  Train Best Model on FULL DATA (TRAIN + TEST)
# ============================================================

best_model.fit(X, y)


# ============================================================
#  Build Full Graph
# ============================================================

all_nodes = set()

for u, v in edges:
    all_nodes.add(u)
    all_nodes.add(v)

G_full = nx.DiGraph()

G_full.add_nodes_from(all_nodes)
G_full.add_edges_from(edges)

in_cache, out_cache, degree_cache, in_deg, out_deg = precompute(G_full)


# ============================================================
#  Extract Test Features (Competition Test)
# ============================================================

test_df = pd.read_csv(
    "../data/test.csv"
)

X_test = []

print("Extracting test features")

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

    u = str(row["From"]).strip()
    v = str(row["To"]).strip()

    X_test.append(
        extract_features(
            u,
            v,
            G_full,
            in_cache,
            out_cache,
            degree_cache,
            in_deg,
            out_deg
        )
    )

X_test = np.array(X_test)


# ============================================================
#  Predict Test Data
# ============================================================

print("Predicting")

probs = best_model.predict_proba(X_test)[:,1]

submission = pd.DataFrame({

    "Id": test_df["Id"],
    "Predictions": probs
})

submission.to_csv("../results/submission.csv", index=False)

print("Submission saved")