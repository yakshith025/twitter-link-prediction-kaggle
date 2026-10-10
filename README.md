# Twitter Link Prediction (Kaggle)

Predict which edges exist in a partial crawl of the Twitter follow graph (**24.0 M directed edges, 4.87 M users**) using 27 hand-engineered graph features and XGBoost. **Best Kaggle score: 0.88.**

## Problem

The test set has 2,000 candidate edges. Half are real follow relationships and half are fabricated. The task is to score each edge's probability of existing.

## Approach

- **Sampling:** 40,000 positive edges and 150,000 negative node pairs.
- **27 graph features per pair:** degrees and in/out ratios, Jaccard, common neighbours, two-hop transitive paths, Adamic-Adar, resource allocation, preferential attachment, degree differences.
- **Models:** Logistic Regression, Random Forest and XGBoost, compared with stratified 5-fold CV (ROC-AUC).

## Results

| Model | CV ROC-AUC |
|---|---|
| XGBoost (final) | ~0.99 |

Kaggle leaderboard progression: 0.60 → 0.85 → 0.86 → **0.88**. Each step is in `code/experiments/` and `results/kaggle_submissions/`.

## Repository structure

```
code/
├── main.py                      final pipeline (features → XGBoost → submission)
├── graph_exploration.ipynb      graph statistics and visualisation
└── experiments/                 numbered iterations, Kaggle scores in file names
data/
├── train.csv                    edge list, ~180 MB (stored with Git LFS)
├── test.csv
└── sample_submission.csv
results/kaggle_submissions/      submitted prediction files
report/final_report.pdf          project report
```

## Getting started

`data/train.csv` is stored with [Git LFS](https://git-lfs.com). Install Git LFS before cloning so the real file is downloaded:

```bash
git lfs install
git clone https://github.com/yakshith025/twitter-link-prediction-kaggle.git
cd twitter-link-prediction-kaggle
pip install -r requirements.txt
cd code
python main.py
```

`main.py` reads `../data/` and writes `../results/submission.csv`.

`code/experiments/06_script_score0.8834_damaged_paste.txt` is a copy of the code that lost characters when it was pasted. It is kept for reference only.

## References

- Cukierski, Hamner & Yang (2011). *Graph-based features for supervised link prediction.* IJCNN.
- Fire et al. (2011). *Link prediction in social networks using computationally efficient topological features.* SocialCom.
- Fire et al. (2013). *Computationally efficient link prediction in a variety of social networks.* ACM TIST.

## Context

CP 218 Theory & Applications of Bayesian Learning, Indian Institute of Science (IISc), Jan–Apr 2026. Individual project by Yakshith Maddineni (Kaggle team "SVG").
