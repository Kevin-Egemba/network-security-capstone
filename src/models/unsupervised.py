"""
Unsupervised anomaly detection and clustering for BETH system-call telemetry.

AnomalyDetector wraps five complementary algorithms:
  - K-Means        : cluster-based (finds dense normal groups)
  - DBSCAN         : density-based (flags low-density outliers)
  - Isolation Forest : tree-based (efficient for high-dimensional anomaly)
  - Gaussian Mixture : probabilistic (soft cluster assignment + log-likelihood)
  - PCA Reconstruction Error : linear subspace (flags points poorly explained)

Usage
-----
    from src.models.unsupervised import AnomalyDetector

    detector = AnomalyDetector()
    results = detector.fit_all(X)                 # fits all 5 algorithms
    preds = detector.predict(X, algorithm="if")   # isolation forest
    report = detector.compare(X, y_true)          # compare vs sparse labels
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.config import settings


class AnomalyDetector:
    """
    Ensemble of unsupervised anomaly detection algorithms.

    Attributes
    ----------
    models : dict
        Fitted algorithm instances keyed by short alias.
    results : dict
        Per-algorithm diagnostics from `fit_all`.
    """

    ALGORITHM_ALIASES = {
        "kmeans": "K-Means",
        "dbscan": "DBSCAN",
        "if": "Isolation Forest",
        "gmm": "Gaussian Mixture",
        "pca": "PCA Reconstruction Error",
    }

    def __init__(
        self,
        n_clusters: int = 5,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 10,
        if_contamination: float = 0.05,
        n_pca_components: int = 10,
    ):
        self.n_clusters = n_clusters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.if_contamination = if_contamination
        self.n_pca_components = n_pca_components

        self.models: Dict = {}
        self.results: Dict = {}
        self._pca_scaler: Optional[StandardScaler] = None
        self._pca_recon_threshold: Optional[float] = None

    # ── Fit all algorithms ────────────────────────────────────────────────────
    def fit_all(self, X: np.ndarray) -> Dict:
        """
        Fit all five algorithms and return a summary dict.

        Parameters
        ----------
        X : np.ndarray — already scaled feature matrix

        Returns
        -------
        dict  keyed by algorithm alias with diagnostic metrics
        """
        logger.info(f"Fitting 5 anomaly detectors on {X.shape}…")

        self.results = {}
        self.results.update(self._fit_kmeans(X))
        self.results.update(self._fit_dbscan(X))
        self.results.update(self._fit_isolation_forest(X))
        self.results.update(self._fit_gmm(X))
        self.results.update(self._fit_pca(X))

        logger.info("All anomaly detectors fitted.")
        return self.results

    # ── K-Means ───────────────────────────────────────────────────────────────
    def _fit_kmeans(self, X: np.ndarray) -> Dict:
        km = KMeans(
            n_clusters=self.n_clusters, init="k-means++",
            n_init=10, random_state=settings.random_seed
        )
        labels = km.fit_predict(X)
        self.models["kmeans"] = km

        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)),
                               random_state=settings.random_seed)
        logger.info(f"  K-Means (k={self.n_clusters}): silhouette={sil:.4f}")
        return {"kmeans": {"labels": labels, "silhouette": round(sil, 4),
                           "n_clusters": self.n_clusters}}

    # ── DBSCAN ────────────────────────────────────────────────────────────────
    def _fit_dbscan(self, X: np.ndarray) -> Dict:
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples,
                    n_jobs=settings.n_jobs)
        labels = db.fit_predict(X)
        self.models["dbscan"] = db

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (labels == -1).mean()
        logger.info(
            f"  DBSCAN (eps={self.dbscan_eps}): "
            f"{n_clusters} clusters, noise={noise_pct:.2%}"
        )
        return {"dbscan": {"labels": labels, "n_clusters": n_clusters,
                           "noise_pct": round(noise_pct, 4)}}

    # ── Isolation Forest ──────────────────────────────────────────────────────
    def _fit_isolation_forest(self, X: np.ndarray) -> Dict:
        iso = IsolationForest(
            n_estimators=200, contamination=self.if_contamination,
            random_state=settings.random_seed, n_jobs=settings.n_jobs
        )
        raw_preds = iso.fit_predict(X)                  # +1 normal, -1 anomaly
        anomaly_flags = (raw_preds == -1).astype(int)
        scores = -iso.score_samples(X)                  # higher = more anomalous
        self.models["if"] = iso

        anomaly_pct = anomaly_flags.mean()
        logger.info(
            f"  Isolation Forest: anomaly_rate={anomaly_pct:.2%}, "
            f"contamination={self.if_contamination}"
        )
        return {"if": {"labels": anomaly_flags, "scores": scores,
                       "anomaly_pct": round(anomaly_pct, 4)}}

    # ── Gaussian Mixture Model ────────────────────────────────────────────────
    def _fit_gmm(self, X: np.ndarray) -> Dict:
        gmm = GaussianMixture(
            n_components=self.n_clusters, covariance_type="full",
            random_state=settings.random_seed
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        log_likelihood = gmm.score(X)                   # per-sample mean log-prob
        # Low log-likelihood → potential anomaly
        sample_ll = gmm.score_samples(X)
        threshold = np.percentile(sample_ll, self.if_contamination * 100)
        anomaly_flags = (sample_ll < threshold).astype(int)
        self.models["gmm"] = gmm
        self._gmm_threshold = threshold

        logger.info(f"  GMM (k={self.n_clusters}): "
                    f"mean_log_likelihood={log_likelihood:.4f}")
        return {"gmm": {"labels": labels, "anomaly_flags": anomaly_flags,
                        "mean_log_likelihood": round(log_likelihood, 4),
                        "ll_threshold": round(threshold, 4)}}

    # ── PCA Reconstruction Error ──────────────────────────────────────────────
    def _fit_pca(self, X: np.ndarray) -> Dict:
        n_comp = min(self.n_pca_components, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=settings.random_seed)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
        recon_error = np.mean((X - X_reconstructed) ** 2, axis=1)

        threshold = np.percentile(recon_error, (1 - self.if_contamination) * 100)
        anomaly_flags = (recon_error > threshold).astype(int)

        self.models["pca"] = pca
        self._pca_recon_threshold = threshold
        variance_explained = pca.explained_variance_ratio_.sum()

        logger.info(
            f"  PCA ({n_comp} components): "
            f"var_explained={variance_explained:.2%}, "
            f"anomaly_rate={anomaly_flags.mean():.2%}"
        )
        return {"pca": {"anomaly_flags": anomaly_flags, "recon_error": recon_error,
                        "variance_explained": round(variance_explained, 4),
                        "threshold": round(threshold, 6), "n_components": n_comp}}

    # ── Predict (new data) ────────────────────────────────────────────────────
    def predict(self, X: np.ndarray, algorithm: str = "if") -> np.ndarray:
        """
        Predict anomaly labels (0=normal, 1=anomaly) for new data.

        Parameters
        ----------
        algorithm : str  'kmeans' | 'dbscan' | 'if' | 'gmm' | 'pca'
        """
        if algorithm not in self.models:
            raise ValueError(f"Algorithm '{algorithm}' not fitted. Call fit_all() first.")

        model = self.models[algorithm]

        if algorithm == "kmeans":
            # Distance to nearest centroid > 95th percentile → anomaly
            distances = np.min(model.transform(X), axis=1)
            threshold = np.percentile(distances, 95)
            return (distances > threshold).astype(int)

        if algorithm == "dbscan":
            return (model.fit_predict(X) == -1).astype(int)

        if algorithm == "if":
            return (model.predict(X) == -1).astype(int)

        if algorithm == "gmm":
            ll = model.score_samples(X)
            return (ll < self._gmm_threshold).astype(int)

        if algorithm == "pca":
            X_rec = model.inverse_transform(model.transform(X))
            err = np.mean((X - X_rec) ** 2, axis=1)
            return (err > self._pca_recon_threshold).astype(int)

        raise ValueError(f"Unknown algorithm: {algorithm}")

    # ── Compare against sparse ground truth ───────────────────────────────────
    def compare(self, X: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all fitted algorithms against sparse ground truth labels.

        Returns a DataFrame with per-algorithm metrics.
        """
        rows = []
        for alias in self.ALGORITHM_ALIASES:
            if alias not in self.models:
                continue
            try:
                y_pred = self.predict(X, alias)
                from sklearn.metrics import precision_score, recall_score, f1_score
                rows.append({
                    "algorithm": self.ALGORITHM_ALIASES[alias],
                    "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                    "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                    "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                    "anomaly_rate": round(y_pred.mean(), 4),
                    "ari": round(adjusted_rand_score(y_true, y_pred), 4),
                })
            except Exception as e:
                logger.warning(f"  {alias} comparison failed: {e}")

        return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    # ── Elbow / BIC selection helpers ─────────────────────────────────────────
    @staticmethod
    def optimal_k_kmeans(X: np.ndarray, k_range: range = range(2, 11)) -> pd.DataFrame:
        """Return inertia and silhouette scores for each k (elbow method)."""
        rows = []
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=settings.random_seed)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels, sample_size=min(3000, len(X)),
                                   random_state=settings.random_seed)
            rows.append({"k": k, "inertia": km.inertia_, "silhouette": round(sil, 4)})
        return pd.DataFrame(rows)

    @staticmethod
    def optimal_k_gmm(X: np.ndarray, k_range: range = range(2, 11)) -> pd.DataFrame:
        """Return BIC and AIC for each number of GMM components."""
        rows = []
        for k in k_range:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=settings.random_seed)
            gmm.fit(X)
            rows.append({"k": k, "bic": round(gmm.bic(X), 2), "aic": round(gmm.aic(X), 2)})
        return pd.DataFrame(rows)
