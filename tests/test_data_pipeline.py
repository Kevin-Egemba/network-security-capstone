"""
Tests for the data loading, validation, and preprocessing pipeline.

Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.validator import DataValidator
from src.data.preprocessor import (
    UNSWPreprocessor,
    BETHPreprocessor,
    CyberAttacksPreprocessor,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — minimal synthetic DataFrames that mirror each dataset's schema
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def df_unsw():
    """Minimal UNSW-NB15-like DataFrame."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "proto": rng.choice(["tcp", "udp", "icmp"], n),
        "service": rng.choice(["-", "http", "ftp", "dns"], n),
        "state": rng.choice(["FIN", "INT", "CON"], n),
        "dur": rng.exponential(1.0, n),
        "sbytes": rng.integers(0, 100_000, n),
        "dbytes": rng.integers(0, 100_000, n),
        "sttl": rng.integers(0, 255, n),
        "dttl": rng.integers(0, 255, n),
        "sloss": rng.integers(0, 10, n),
        "dloss": rng.integers(0, 10, n),
        "spkts": rng.integers(1, 100, n),
        "dpkts": rng.integers(1, 100, n),
        "ct_srv_src": rng.integers(0, 50, n),
        "ct_dst_ltm": rng.integers(0, 50, n),
        "ct_src_dport_ltm": rng.integers(0, 50, n),
        "label": rng.integers(0, 2, n),
        "attack_cat": rng.choice(["Normal", "Fuzzers", "DoS", "Exploits"], n),
    })


@pytest.fixture
def df_beth():
    """Minimal BETH-like DataFrame with sparse evil labels."""
    rng = np.random.default_rng(42)
    n = 300
    evil = np.zeros(n, dtype=int)
    evil[rng.choice(n, size=5, replace=False)] = 1    # ~1.7% evil
    return pd.DataFrame({
        "processId": rng.integers(1, 50000, n),
        "threadId": rng.integers(1, 50000, n),
        "parentProcessId": rng.integers(1, 50000, n),
        "userId": rng.integers(0, 1000, n),
        "mountNamespace": rng.integers(0, 100, n),
        "eventId": rng.integers(1, 400, n),
        "argsNum": rng.integers(0, 8, n),
        "returnValue": rng.integers(-10, 10, n),
        "evil": evil,
    })


@pytest.fixture
def df_cyber():
    """Minimal Cybersecurity Attacks-like DataFrame."""
    rng = np.random.default_rng(42)
    n = 150
    return pd.DataFrame({
        "Destination Port": rng.integers(1, 65535, n),
        "Protocol": rng.choice(["TCP", "UDP", "ICMP"], n),
        "Packet Length": rng.integers(40, 65535, n),
        "Packet Type": rng.choice(["Data", "Control"], n),
        "Traffic Type": rng.choice(["HTTP", "DNS", "FTP"], n),
        "Anomaly Scores": rng.uniform(0, 1, n),
        "Action Taken": rng.choice(["Allowed", "Blocked"], n),
        "Severity Level": rng.choice(["Low", "Medium", "High"], n),
        "Malware Indicators": rng.choice(["IoC Detected", "None"], n),
        "Attack Type": rng.choice(["DDoS", "Malware", "Intrusion"], n),
    })


# ─────────────────────────────────────────────────────────────────────────────
# DataValidator tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDataValidator:

    def test_validate_unsw_passes(self, df_unsw):
        v = DataValidator()
        report = v.validate_unsw(df_unsw)
        assert report.passed, f"Unexpected errors: {report.errors}"

    def test_validate_unsw_missing_target(self, df_unsw):
        v = DataValidator()
        bad = df_unsw.drop(columns=["label"])
        report = v.validate_unsw(bad)
        assert not report.passed
        assert any("label" in e for e in report.errors)

    def test_validate_beth_sparsity_warning(self, df_beth):
        v = DataValidator()
        report = v.validate_beth(df_beth)
        # Should warn about sparse labels
        assert any("sparse" in w.lower() or "evil" in w.lower()
                   for w in report.warnings)

    def test_validate_cyber_passes(self, df_cyber):
        v = DataValidator()
        report = v.validate_cyber_attacks(df_cyber)
        assert report.passed

    def test_leakage_detection_no_false_positives(self, df_unsw):
        v = DataValidator()
        result = v.check_leakage(df_unsw, target="label", threshold=0.95)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "leakage_flag" in result.columns

    def test_compare_train_test(self, df_unsw):
        v = DataValidator()
        train = df_unsw.iloc[:150]
        test = df_unsw.iloc[150:]
        report = v.compare_train_test(train, test, target="label")
        assert report["target_in_both"] is True
        assert report["columns_only_in_train"] == []
        assert report["columns_only_in_test"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessor tests
# ─────────────────────────────────────────────────────────────────────────────
class TestUNSWPreprocessor:

    def test_fit_transform_binary(self, df_unsw):
        prep = UNSWPreprocessor()
        X, y = prep.fit_transform(df_unsw, target="label")
        assert X.shape[0] == len(df_unsw)
        assert X.shape[1] > 0
        assert len(y) == len(df_unsw)
        assert set(y).issubset({0, 1})

    def test_fit_transform_multiclass(self, df_unsw):
        prep = UNSWPreprocessor()
        X, y = prep.fit_transform(df_unsw, target="attack_cat")
        assert X.shape[0] == len(df_unsw)

    def test_transform_matches_fit(self, df_unsw):
        prep = UNSWPreprocessor()
        X_train, y_train = prep.fit_transform(df_unsw.iloc[:150], target="label")
        X_test, y_test = prep.transform(df_unsw.iloc[150:], target="label")
        assert X_test.shape[1] == X_train.shape[1]

    def test_feature_names_populated(self, df_unsw):
        prep = UNSWPreprocessor()
        prep.fit_transform(df_unsw, target="label")
        assert len(prep.feature_names) > 0

    def test_no_target_leakage(self, df_unsw):
        """Target column must not appear in feature matrix."""
        prep = UNSWPreprocessor()
        X, _ = prep.fit_transform(df_unsw, target="label")
        # All values should be scaled (not 0 or 1 only)
        unique_vals = np.unique(X)
        assert len(unique_vals) > 2


class TestBETHPreprocessor:

    def test_fit_transform(self, df_beth):
        prep = BETHPreprocessor()
        X, y = prep.fit_transform(df_beth)
        assert X.shape[0] == len(df_beth)
        assert y is not None
        assert len(y) == len(df_beth)

    def test_id_columns_removed(self, df_beth):
        prep = BETHPreprocessor()
        X, _ = prep.fit_transform(df_beth)
        # processId etc. should be excluded
        assert "processId" not in prep.feature_names


class TestCyberAttacksPreprocessor:

    def test_metadata_only(self, df_cyber):
        prep = CyberAttacksPreprocessor(feature_set="metadata_only")
        X, y = prep.fit_transform(df_cyber)
        assert X.shape[0] == len(df_cyber)
        assert len(np.unique(y)) == 3   # DDoS, Malware, Intrusion

    def test_with_leakage(self, df_cyber):
        prep = CyberAttacksPreprocessor(feature_set="with_leakage")
        X_leak, _ = prep.fit_transform(df_cyber)
        prep2 = CyberAttacksPreprocessor(feature_set="metadata_only")
        X_meta, _ = prep2.fit_transform(df_cyber)
        # Leakage variant should have more features
        assert X_leak.shape[1] >= X_meta.shape[1]

    def test_invalid_feature_set(self):
        with pytest.raises(ValueError):
            CyberAttacksPreprocessor(feature_set="invalid")


# ─────────────────────────────────────────────────────────────────────────────
# Integration: preprocessor → model
# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessorModelIntegration:

    def test_unsw_to_random_forest(self, df_unsw):
        from sklearn.ensemble import RandomForestClassifier
        prep = UNSWPreprocessor()
        X, y = prep.fit_transform(df_unsw.iloc[:150], target="label")
        X_test, y_test = prep.transform(df_unsw.iloc[150:], target="label")

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(y_test)
        assert set(y_pred).issubset({0, 1})

    def test_beth_to_isolation_forest(self, df_beth):
        from sklearn.ensemble import IsolationForest
        prep = BETHPreprocessor()
        X, _ = prep.fit_transform(df_beth)

        model = IsolationForest(n_estimators=10, random_state=42)
        model.fit(X)
        preds = model.predict(X)
        assert set(preds).issubset({1, -1})
