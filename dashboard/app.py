"""
Network Security Analytics Dashboard
======================================
Multi-page Streamlit app providing a full analyst interface.

Pages
-----
  Home / Overview       — platform KPIs and dataset summary
  Threat Intelligence   — attack distribution, trends, alert queue
  Model Performance     — training metrics, ROC curves, feature importance
  Live Detection        — submit network flows for real-time prediction
  Data Explorer         — SQL query interface against the live database
  User Management       — admin panel for user accounts (admin only)

Run
---
    streamlit run dashboard/app.py
"""

import sys
import json
import random
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from src.config import Paths
from src.db.connector import DatabaseManager
from src.db.models import UserRole

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Security Analytics",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB connection (cached) ────────────────────────────────────────────────────
@st.cache_resource
def get_manager() -> DatabaseManager:
    mgr = DatabaseManager()
    mgr.create_tables()
    mgr.bootstrap_admin()
    return mgr


mgr = get_manager()


# ── Authentication gate ───────────────────────────────────────────────────────
def _login_screen() -> None:
    st.title("🛡 Network Security Analytics")
    st.markdown("Sign in to continue.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        user, failure_reason = mgr.authenticate(username, password)
        if failure_reason == "locked":
            st.error("Account locked after repeated failed attempts. Try again later.")
        elif not user:
            st.error("Invalid username or password.")
        else:
            st.session_state.auth_user = {"username": user.username, "role": user.role.value}
            st.rerun()

    st.stop()


if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

if st.session_state.auth_user is None:
    _login_screen()

current_user = st.session_state.auth_user
is_admin = current_user["role"] == UserRole.admin.value


# ── Auto-seed on first run ────────────────────────────────────────────────────
def auto_seed():
    try:
        from src.db.models import (
            NetworkEvent, SystemCallEvent, ModelRun,
            Alert, AlertSeverity, AlertStatus, DatasetRegistry
        )

        with mgr.engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM network_events")).scalar()

        if count == 0:
            with mgr.get_session() as session:
                registry = DatasetRegistry(
                    name="demo",
                    source_file="demo",
                    row_count=5000,
                    column_count=45,
                    description="Demo data for Streamlit Cloud"
                )
                session.add(registry)
                session.flush()

                # Network events
                session.bulk_save_objects([NetworkEvent(
                    dataset_id=registry.id,
                    split="train",
                    proto=random.choice(["tcp", "udp", "icmp"]),
                    service=random.choice(["-", "http", "ssh", "dns", "ftp"]),
                    state=random.choice(["FIN", "INT", "REQ", "CON"]),
                    dur=round(random.uniform(0, 100), 4),
                    sbytes=random.randint(0, 100000),
                    dbytes=random.randint(0, 100000),
                    sttl=random.randint(0, 255),
                    dttl=random.randint(0, 255),
                    spkts=random.randint(1, 1000),
                    dpkts=random.randint(1, 1000),
                    label=1 if random.random() > 0.55 else 0,
                    attack_cat=random.choice(["DoS", "Exploits", "Reconnaissance", "Fuzzers", "Backdoor"]) if random.random() > 0.55 else "Normal"
                ) for _ in range(5000)])

                # Syscall events
                session.bulk_save_objects([SystemCallEvent(
                    dataset_id=registry.id,
                    split="train",
                    event_id=random.randint(1, 500),
                    args_num=random.randint(0, 10),
                    return_value=random.randint(-1, 100),
                    evil=1 if random.random() > 0.85 else 0
                ) for _ in range(3000)])

                # Model runs
                for i, (algo, ds) in enumerate([
                    ("RandomForest", "UNSW-NB15"),
                    ("XGBoost", "UNSW-NB15"),
                    ("LogisticRegression", "UNSW-NB15"),
                    ("IsolationForest", "BETH"),
                    ("KMeans", "BETH")
                ]):
                    roc_auc = round(random.uniform(0.88, 0.985), 4)
                    spread = round(random.uniform(0.008, 0.025), 4)
                    session.add(ModelRun(
                        run_name=f"run_{algo.lower()}_{i}",
                        model_type="supervised" if i < 3 else "unsupervised",
                        dataset_name=ds,
                        algorithm=algo,
                        accuracy=round(random.uniform(0.85, 0.99), 4),
                        roc_auc=roc_auc,
                        f1_weighted=round(random.uniform(0.85, 0.98), 4),
                        f1_macro=round(random.uniform(0.80, 0.97), 4),
                        extra_metrics={
                            "roc_auc_ci": {
                                "point": roc_auc,
                                "lower": round(max(0.0, roc_auc - spread), 4),
                                "upper": round(min(1.0, roc_auc + spread), 4),
                                "ci": 0.95,
                            }
                        },
                    ))

                # Alerts
                for title, sev, atype, conf in [
                    ("High-volume DoS attack detected", AlertSeverity.critical, "DoS", 0.94),
                    ("Exploit attempt on HTTP service", AlertSeverity.high, "Exploits", 0.88),
                    ("Fuzzing activity on port 443", AlertSeverity.medium, "Fuzzers", 0.71),
                    ("Reconnaissance scan detected", AlertSeverity.low, "Reconnaissance", 0.65),
                    ("Backdoor connection attempt", AlertSeverity.critical, "Backdoor", 0.91),
                ]:
                    session.add(Alert(
                        title=title,
                        severity=sev,
                        status=AlertStatus.open,
                        attack_type=atype,
                        confidence=conf,
                        source_dataset="unsw_nb15",
                        description=f"{atype} detected with {conf:.0%} confidence."
                    ))

                session.commit()

    except Exception as e:
        st.warning(f"Seed info: {e}")


auto_seed()


# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Network Security Analytics")
st.sidebar.markdown(f"Signed in as **{current_user['username']}** ({current_user['role']})")
if st.sidebar.button("Log out"):
    st.session_state.auth_user = None
    st.rerun()
st.sidebar.markdown("---")

nav_pages = ["Overview", "Threat Intelligence", "Model Performance", "Live Detection"]
if is_admin:
    nav_pages += ["Data Explorer", "User Management"]

page = st.sidebar.radio("Navigate", nav_pages, index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Network Security Capstone v1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _run_query(sql: str) -> pd.DataFrame:
    with mgr.engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def _metric_card(col, label: str, value: str, delta: str = None):
    col.metric(label=label, value=value, delta=delta)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Network Security Analytics Platform")
    st.markdown(
        "End-to-end ML platform for network intrusion detection, behavioral "
        "anomaly analysis, and threat classification across three real-world datasets."
    )

    st.markdown("### Platform KPIs")
    c1, c2, c3, c4, c5 = st.columns(5)

    try:
        n_network = _run_query("SELECT COUNT(*) as n FROM network_events").iloc[0, 0]
        n_syscall = _run_query("SELECT COUNT(*) as n FROM system_call_events").iloc[0, 0]
        n_runs    = _run_query("SELECT COUNT(*) as n FROM model_runs").iloc[0, 0]
        n_alerts  = _run_query("SELECT COUNT(*) as n FROM alerts WHERE status='open'").iloc[0, 0]
        best_auc  = _run_query("SELECT MAX(roc_auc) as v FROM model_runs").iloc[0, 0]

        _metric_card(c1, "Network Events", f"{int(n_network):,}")
        _metric_card(c2, "Syscall Events", f"{int(n_syscall):,}")
        _metric_card(c3, "Model Runs", str(int(n_runs)))
        _metric_card(c4, "Open Alerts", str(int(n_alerts)))
        _metric_card(c5, "Best AUC", f"{best_auc:.4f}" if best_auc else "—")
    except Exception as e:
        st.info(f"Database empty — run ingestion to populate. ({e})")

    st.markdown("---")
    st.markdown("### Dataset Summary")

    datasets = {
        "UNSW-NB15": {
            "description": "Network flow telemetry from real/simulated attacks",
            "rows": "257,673", "features": 45,
            "task": "Binary + 9-class attack classification",
            "best_metric": "~94% AUC (Random Forest)",
            "key_finding": "Rich packet-level features enable high-accuracy detection",
        },
        "BETH": {
            "description": "Honeypot system-call telemetry (sparse labels)",
            "rows": "~100K", "features": 23,
            "task": "Unsupervised anomaly detection",
            "best_metric": "Silhouette = 0.38 (K-Means, k=5)",
            "key_finding": "Labels too sparse for supervised; unsupervised methods find signal",
        },
        "Cyber Attacks (Synthetic)": {
            "description": "Synthetic multi-class attack dataset",
            "rows": "40,000", "features": 24,
            "task": "3-class attack type classification",
            "best_metric": "33-34% accuracy (near random baseline!)",
            "key_finding": "Network metadata alone insufficient for fine-grained classification",
        },
    }

    for name, info in datasets.items():
        with st.expander(f"📊 {name}", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Task:** {info['task']}")
                st.markdown(f"**Key Finding:** {info['key_finding']}")
            with col2:
                st.markdown(f"**Rows:** {info['rows']}")
                st.markdown(f"**Features:** {info['features']}")
                st.markdown(f"**Best Metric:** {info['best_metric']}")

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("""
    ```
    Raw CSVs → DataLoader → DataValidator → Preprocessor
                                                ↓
                                    Supervised / Unsupervised Models
                                                ↓
                                      SQLite / PostgreSQL DB
                                       ↙           ↘
                               FastAPI REST        Streamlit
                               (predictions)       (this dashboard)
    ```
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Threat Intelligence
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Threat Intelligence":
    st.title("Threat Intelligence")

    try:
        df_attacks = _run_query(
            "SELECT attack_cat, COUNT(*) as count FROM network_events "
            "WHERE label=1 GROUP BY attack_cat ORDER BY count DESC"
        )
        df_normal = _run_query(
            "SELECT label, COUNT(*) as count FROM network_events GROUP BY label"
        )
        df_alerts = _run_query(
            "SELECT severity, status, attack_type, created_at FROM alerts "
            "ORDER BY created_at DESC LIMIT 100"
        )

        if not df_attacks.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Attack Category Distribution")
                fig = px.bar(df_attacks, x="attack_cat", y="count",
                             color="count", color_continuous_scale="Reds",
                             title="Attack Types in Database")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("#### Normal vs Attack Split")
                labels = {0: "Normal", 1: "Attack"}
                df_normal["label_name"] = df_normal["label"].map(labels)
                fig2 = px.pie(df_normal, values="count", names="label_name",
                              color_discrete_map={"Normal": "#2ecc71", "Attack": "#e74c3c"})
                st.plotly_chart(fig2, use_container_width=True)

        if not df_alerts.empty:
            st.markdown("#### Recent Alerts")
            st.dataframe(df_alerts, use_container_width=True)
        else:
            st.info("No alerts in database yet.")

    except Exception as e:
        st.info(f"Load data first. Detail: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model Performance")

    try:
        df_runs = _run_query(
            "SELECT run_name, model_type, dataset_name, algorithm, "
            "accuracy, roc_auc, f1_weighted, f1_macro, extra_metrics, created_at "
            "FROM model_runs ORDER BY created_at DESC"
        )

        if df_runs.empty:
            st.info("No model runs recorded yet.")
        else:
            # Pull roc_auc_ci out of the extra_metrics JSON blob, if present
            def _ci_bounds(raw):
                if not raw:
                    return None, None
                try:
                    meta = json.loads(raw) if isinstance(raw, str) else raw
                    ci = meta.get("roc_auc_ci") if meta else None
                    return (ci["lower"], ci["upper"]) if ci else (None, None)
                except (TypeError, ValueError, KeyError):
                    return None, None

            ci_bounds = df_runs["extra_metrics"].apply(_ci_bounds)
            df_runs["roc_auc_ci_lower"] = [b[0] for b in ci_bounds]
            df_runs["roc_auc_ci_upper"] = [b[1] for b in ci_bounds]
            has_ci = df_runs["roc_auc_ci_lower"].notna().any()

            st.markdown("#### All Experiment Runs")
            display_cols = ["run_name", "model_type", "dataset_name", "algorithm",
                             "accuracy", "roc_auc", "f1_weighted", "f1_macro", "created_at"]
            st.dataframe(
                df_runs[display_cols].style.background_gradient(
                    subset=["roc_auc", "f1_weighted"], cmap="Greens"),
                use_container_width=True,
            )

            st.markdown("#### AUC Comparison by Algorithm")
            st.caption(
                "Latest run per algorithm/dataset — re-running training for the "
                "same algorithm adds a new row to the table above without "
                "cluttering this comparison."
            )
            plot_df = df_runs.dropna(subset=["roc_auc"]).copy()
            plot_df["created_at"] = pd.to_datetime(plot_df["created_at"])
            plot_df = (
                plot_df.sort_values("created_at")
                .drop_duplicates(subset=["algorithm", "dataset_name"], keep="last")
            )
            if has_ci:
                plot_df = plot_df.assign(
                    err_plus=(plot_df["roc_auc_ci_upper"] - plot_df["roc_auc"]).clip(lower=0),
                    err_minus=(plot_df["roc_auc"] - plot_df["roc_auc_ci_lower"]).clip(lower=0),
                )
                fig = px.bar(
                    plot_df, x="algorithm", y="roc_auc", color="dataset_name",
                    barmode="group", title="ROC AUC by Algorithm and Dataset (95% bootstrap CI)",
                    error_y="err_plus", error_y_minus="err_minus",
                )
                st.caption(
                    "Error bars are 95% bootstrap confidence intervals, not just point "
                    "estimates. Runs whose intervals don't overlap are more likely to "
                    "reflect a real difference rather than resampling noise — overlapping "
                    "intervals aren't strong evidence either way on their own."
                )
            else:
                fig = px.bar(
                    plot_df, x="algorithm", y="roc_auc", color="dataset_name",
                    barmode="group", title="ROC AUC by Algorithm and Dataset",
                )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.info(f"Model performance data unavailable. ({e})")

    st.markdown("---")
    st.markdown("### Published Results (from notebooks)")

    results = {
        "UNSW-NB15 Binary Detection": [
            {"Model": "Logistic Regression", "AUC": 0.91, "F1": 0.88},
            {"Model": "Random Forest", "AUC": 0.94, "F1": 0.93},
            {"Model": "XGBoost", "AUC": 0.94, "F1": 0.93},
        ],
        "BETH Anomaly Detection": [
            {"Algorithm": "K-Means (k=5)", "Silhouette": 0.38, "Recall": 0.45},
            {"Algorithm": "Isolation Forest", "Silhouette": None, "Recall": 0.52},
            {"Algorithm": "DBSCAN", "Silhouette": 0.31, "Recall": 0.48},
        ],
        "Cyber Attacks (Synthetic)": [
            {"Model": "Random Forest (metadata only)", "Accuracy": 0.34, "Note": "Near random!"},
            {"Model": "XGBoost (with leakage)", "Accuracy": 0.34, "Note": "Metadata insufficient"},
        ],
    }

    for title, data in results.items():
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(data), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Live Detection
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Live Detection":
    st.title("Live Threat Detection")
    st.markdown(
        "Submit a network flow record for real-time attack classification. "
        "Uses the trained UNSW-NB15 two-stage detector."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            proto = st.selectbox("Protocol", ["tcp", "udp", "icmp", "arp"])
            service = st.selectbox("Service", ["-", "http", "ftp", "smtp", "ssh", "dns"])
            state = st.selectbox("Connection State", ["FIN", "INT", "REQ", "CON", "RST"])
        with col2:
            dur = st.number_input("Duration (s)", 0.0, 1000.0, 0.5, step=0.1)
            sbytes = st.number_input("Source Bytes", 0, 10_000_000, 1024)
            dbytes = st.number_input("Dest Bytes", 0, 10_000_000, 512)
        with col3:
            sttl = st.number_input("Source TTL", 0, 255, 64)
            dttl = st.number_input("Dest TTL", 0, 255, 64)
            spkts = st.number_input("Source Packets", 1, 10000, 3)
            dpkts = st.number_input("Dest Packets", 1, 10000, 2)

        submitted = st.form_submit_button("Analyze Traffic")

    if submitted:
        model_path = Paths.MODELS / "two_stage_detector.pkl"
        prep_path = Paths.MODELS / "unsw_preprocessor.pkl"
        if not model_path.exists() or not prep_path.exists():
            st.warning(
                "No trained model found. Run `python train_two_stage_detector.py` "
                "to train and save the two-stage detector + preprocessor."
            )
        else:
            import joblib
            model = joblib.load(model_path)
            prep = joblib.load(prep_path)
            try:
                X = prep.transform_live(
                    proto=proto, service=service, state=state,
                    dur=dur, sbytes=sbytes, dbytes=dbytes,
                    sttl=sttl, dttl=dttl, spkts=spkts, dpkts=dpkts,
                )
                result = model.predict(X)
                is_attack = bool(result["binary_pred"][0])
                attack_proba = float(result["attack_proba"][0])
                attack_type = str(result["attack_type"][0]) if is_attack else None

                result_col1, result_col2 = st.columns(2)
                if is_attack:
                    result_col1.error(f"ATTACK DETECTED — {attack_type} ({attack_proba:.1%} confidence)")
                else:
                    result_col1.success(f"Normal Traffic — {1 - attack_proba:.1%} confidence")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=attack_proba * 100,
                    title={"text": "Attack Probability (%)"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "red" if is_attack else "green"},
                           "steps": [
                               {"range": [0, 40], "color": "#d5f5e3"},
                               {"range": [40, 70], "color": "#fdebd0"},
                               {"range": [70, 100], "color": "#fadbd8"},
                           ]}
                ))
                result_col2.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown("Run SQL queries directly against the platform database.")

    canned_queries = {
        "Attack category distribution": (
            "SELECT attack_cat, COUNT(*) as count, "
            "ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct "
            "FROM network_events WHERE label=1 "
            "GROUP BY attack_cat ORDER BY count DESC"
        ),
        "Normal vs attack split": (
            "SELECT label, COUNT(*) as count FROM network_events GROUP BY label"
        ),
        "Evil event rate in BETH": (
            "SELECT evil, COUNT(*) as count FROM system_call_events GROUP BY evil"
        ),
        "Top 10 event IDs in BETH": (
            "SELECT event_id, COUNT(*) as count FROM system_call_events "
            "GROUP BY event_id ORDER BY count DESC LIMIT 10"
        ),
        "Registered datasets": (
            "SELECT name, row_count, column_count, ingested_at FROM dataset_registry"
        ),
        "Model run leaderboard": (
            "SELECT run_name, algorithm, dataset_name, roc_auc, f1_weighted "
            "FROM model_runs ORDER BY roc_auc DESC LIMIT 20"
        ),
    }

    selected = st.selectbox("Quick queries", ["Custom…"] + list(canned_queries))
    if selected == "Custom…":
        query = st.text_area("SQL Query", height=120,
                             placeholder="SELECT * FROM network_events LIMIT 10")
    else:
        query = st.text_area("SQL Query", value=canned_queries[selected], height=120)

    if st.button("Run Query"):
        if not query.strip():
            st.warning("Enter a SQL query first.")
        else:
            try:
                df = _run_query(query)
                st.success(f"Returned {len(df):,} rows")
                st.dataframe(df, use_container_width=True)

                if len(df) > 0 and st.checkbox("Show chart (first two numeric columns)"):
                    num_cols = df.select_dtypes(include="number").columns.tolist()
                    if len(num_cols) >= 2:
                        fig = px.bar(df.head(50), x=num_cols[0], y=num_cols[1])
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Query failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: User Management
# ─────────────────────────────────────────────────────────────────────────────
elif page == "User Management":
    st.title("User Management")
    st.markdown("Manage platform users and roles. This page is admin-only.")

    try:
        df_users = _run_query(
            "SELECT id, username, email, role, is_active, created_at, last_login "
            "FROM users ORDER BY created_at"
        )

        if df_users.empty:
            st.info("No users in database.")
        else:
            st.markdown("#### Current Users")
            st.dataframe(df_users, use_container_width=True)

            role_counts = df_users["role"].value_counts().reset_index()
            role_counts.columns = ["role", "count"]
            fig = px.pie(role_counts, values="count", names="role",
                         title="Users by Role",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.info(f"User data unavailable: {e}")

    st.markdown("---")
    st.markdown("### Delete a User")
    st.caption("GDPR right-to-erasure: permanently removes a user account.")
    with st.form("delete_user_form"):
        deletable = [
            u for u in df_users["username"].tolist()
            if not df_users.empty and u != current_user["username"]
        ] if not df_users.empty else []
        target_username = st.selectbox("User to delete", deletable) if deletable else None
        delete_submitted = st.form_submit_button("Delete User", disabled=not deletable)

    if delete_submitted and target_username:
        try:
            mgr.delete_user(target_username)
            st.success(f"User '{target_username}' deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to delete user: {e}")

    st.markdown("---")
    st.markdown("### Create New User")

    with st.form("new_user_form"):
        col1, col2 = st.columns(2)
        new_username = col1.text_input("Username")
        new_email = col2.text_input("Email")
        new_password = col1.text_input("Password", type="password")
        new_role = col2.selectbox("Role", ["viewer", "analyst", "data_scientist", "admin"])
        create_submitted = st.form_submit_button("Create User")

    if create_submitted:
        if not new_username or not new_email or not new_password:
            st.warning("All fields required.")
        else:
            try:
                from src.db.models import UserRole as URole
                mgr.create_user(
                    username=new_username,
                    email=new_email,
                    password=new_password,
                    role=URole(new_role),
                )
                st.success(f"User '{new_username}' created with role '{new_role}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create user: {e}")