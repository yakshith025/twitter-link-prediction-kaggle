# ============================================================
# IMPORTS
# ============================================================

import csv
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

random.seed(42)
np.random.seed(42)


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
# 2️⃣ Easy Negative Sampling
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
# 3️⃣ Precompute Graph Stats
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
# 4️⃣ FULL 38 FEATURE EXTRACTION
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
    sqrt_weighted = 0

    for z in common:
        deg_z = degree_cache.get(z, 0)
        if deg_z > 1:
            adamic += 1 / np.log(deg_z)
        if deg_z > 0:
            resource += 1 / deg_z
            sqrt_weighted += 1 / np.sqrt(deg_z)

    dice = (2 * cn_count) / (len(u_all) + len(v_all)) if (len(u_all) + len(v_all)) else 0
    salton = cn_count / np.sqrt(u_d * v_d) if u_d > 0 and v_d > 0 else 0
    hpi = cn_count / min(u_d, v_d) if min(u_d, v_d) > 0 else 0
    hdi = cn_count / max(u_d, v_d) if max(u_d, v_d) > 0 else 0
    lhn = cn_count / (u_d * v_d) if u_d > 0 and v_d > 0 else 0

    deg_ratio = u_d / v_d if v_d > 0 else 0
    in_ratio = u_d_in / v_d_in if v_d_in > 0 else 0
    out_ratio = u_d_out / v_d_out if v_d_out > 0 else 0

    log_u_d = np.log1p(u_d)
    log_v_d = np.log1p(v_d)

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
        out_degrees.get(u, 0),
        dice, salton, hpi, hdi, lhn,
        deg_ratio, in_ratio, out_ratio,
        log_u_d, log_v_d,
        sqrt_weighted
    ]


# ============================================================
# 5️⃣ Build Dataset
# ============================================================

def build_training_dataset(train_edges_full):

    positives = random.sample(train_edges_full, 50000)
    remaining_edges = list(set(train_edges_full) - set(positives))

    all_nodes = set()
    for u, v in train_edges_full:
        all_nodes.add(u)
        all_nodes.add(v)

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(remaining_edges)

    negatives = sample_easy_negatives(G, 150000)

    in_cache, out_cache, degree_cache, in_deg, out_deg = precompute(G)

    X, y = [], []

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
# 6️⃣ Load Data
# ============================================================

edges = parse_adjacency_list("/kaggle/input/datasets/yakshithm/project-1/train.csv")

train_edges, _ = train_test_split(edges, test_size=0.2, random_state=42)

X, y = build_training_dataset(train_edges)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scale_pos_weight = 150000 / 50000


# ============================================================
# 7️⃣ Models
# ============================================================

models = {

    "XGBoost_GPU": (
        XGBClassifier(
            tree_method="hist",
            device="cuda",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        ),
        {
            "n_estimators":[700,900],
            "max_depth":[6,8,10],
            "learning_rate":[0.03,0.05],
            "subsample":[0.7,0.8],
            "colsample_bytree":[0.7,0.8]
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators":[300,500],
            "learning_rate":[0.5,1.0]
        }
    ),

    "Bayesian_Logistic": (
        Pipeline([
            ("scaler",StandardScaler()),
            ("logreg",LogisticRegression(max_iter=1000))
        ]),
        {
            "logreg__C":[0.01,0.1,1,10]
        }
    ),

    "GaussianNB": (
        GaussianNB(),
        {
            "var_smoothing":[1e-9,1e-8,1e-7]
        }
    ),

    "BernoulliNB": (
        Pipeline([
            ("scaler",MinMaxScaler()),
            ("bnb",BernoulliNB())
        ]),
        {
            "bnb__alpha":[0.1,0.5,1.0],
            "bnb__binarize":[0.0,0.5]
        }
    )
}


# ============================================================
# 8️⃣ Hyperparameter Tuning
# ============================================================

results = {}

for name,(model,params) in models.items():

    print("\nTuning",name)

    search = RandomizedSearchCV(
        model,
        params,
        n_iter=6,
        scoring="roc_auc",
        cv=skf,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X,y)

    print("Best AUC:",search.best_score_)
    print("Best Params:",search.best_params_)

    results[name] = search.best_estimator_


# ============================================================
# 9️⃣ Final Training
# ============================================================

for name in results:
    results[name].fit(X,y)


# ============================================================
# 🔟 Build Graph for Test
# ============================================================

G_full = nx.DiGraph()
G_full.add_edges_from(edges)

in_cache,out_cache,degree_cache,in_deg,out_deg = precompute(G_full)

test_df = pd.read_csv("/kaggle/input/datasets/yakshithm/project-1/test.csv")

X_test=[]

for _,row in tqdm(test_df.iterrows(),total=len(test_df)):

    X_test.append(
        extract_features(
            str(row["From"]).strip(),
            str(row["To"]).strip(),
            G_full,
            in_cache,out_cache,
            degree_cache,
            in_deg,out_deg
        )
    )

X_test=np.array(X_test)


# ============================================================
# 1️⃣1️⃣ Ensemble Prediction
# ============================================================

probs_xgb = results["XGBoost_GPU"].predict_proba(X_test)[:,1]
probs_ada = results["AdaBoost"].predict_proba(X_test)[:,1]
probs_log = results["Bayesian_Logistic"].predict_proba(X_test)[:,1]
probs_gnb = results["GaussianNB"].predict_proba(X_test)[:,1]
probs_bnb = results["BernoulliNB"].predict_proba(X_test)[:,1]

final_probs = (
    0.45*probs_xgb +
    0.20*probs_ada +
    0.15*probs_log +
    0.10*probs_gnb +
    0.10*probs_bnb
)

submission = pd.DataFrame({
    "Id":test_df["Id"],
    "Predictions":final_probs
})

submission.to_csv("submission1.csv",index=False)

print("Submission saved successfully")