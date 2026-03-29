-- ─────────────────────────────────────────────────────────────────────────────
-- Network Security Analytics Platform — Database Schema
-- Compatible with: SQLite (dev) and PostgreSQL (production)
-- Generated from: src/db/models.py (SQLAlchemy ORM definitions)
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Users ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER      PRIMARY KEY AUTOINCREMENT,
    username        VARCHAR(50)  NOT NULL UNIQUE,
    email           VARCHAR(120) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role            VARCHAR(20)  NOT NULL DEFAULT 'viewer',   -- admin|analyst|data_scientist|viewer
    is_active       BOOLEAN      NOT NULL DEFAULT 1,
    created_at      DATETIME     DEFAULT CURRENT_TIMESTAMP,
    last_login      DATETIME
);
CREATE INDEX IF NOT EXISTS ix_users_username ON users (username);
CREATE INDEX IF NOT EXISTS ix_users_email    ON users (email);

-- ── Dataset Registry ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dataset_registry (
    id              INTEGER      PRIMARY KEY AUTOINCREMENT,
    name            VARCHAR(100) NOT NULL UNIQUE,
    source_file     VARCHAR(500),
    row_count       INTEGER,
    column_count    INTEGER,
    ingested_at     DATETIME     DEFAULT CURRENT_TIMESTAMP,
    schema_version  VARCHAR(20)  DEFAULT '1.0.0',
    description     TEXT,
    extra_metadata  TEXT         -- JSON blob
);

-- ── Network Events (UNSW-NB15) ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS network_events (
    id                  INTEGER  PRIMARY KEY AUTOINCREMENT,
    dataset_id          INTEGER  REFERENCES dataset_registry(id),
    split               VARCHAR(10),                -- train | test
    proto               VARCHAR(20),
    service             VARCHAR(20),
    state               VARCHAR(20),
    dur                 REAL,
    sbytes              INTEGER,
    dbytes              INTEGER,
    sttl                INTEGER,
    dttl                INTEGER,
    sloss               INTEGER,
    dloss               INTEGER,
    spkts               INTEGER,
    dpkts               INTEGER,
    ct_srv_src          INTEGER,
    ct_dst_ltm          INTEGER,
    ct_src_dport_ltm    INTEGER,
    label               INTEGER,                    -- 0=normal, 1=attack
    attack_cat          VARCHAR(30),                -- Fuzzers|DoS|Exploits|...
    ingested_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_network_label     ON network_events (label);
CREATE INDEX IF NOT EXISTS ix_network_label_cat ON network_events (label, attack_cat);

-- ── System Call Events (BETH) ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_call_events (
    id                  INTEGER  PRIMARY KEY AUTOINCREMENT,
    dataset_id          INTEGER  REFERENCES dataset_registry(id),
    split               VARCHAR(20),
    process_id          INTEGER,
    thread_id           INTEGER,
    parent_process_id   INTEGER,
    user_id             INTEGER,
    mount_namespace     INTEGER,
    event_id            INTEGER,
    args_num            INTEGER,
    return_value        INTEGER,
    evil                INTEGER,                    -- 0=normal, 1=evil (sparse)
    ingested_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_syscall_evil       ON system_call_events (evil);
CREATE INDEX IF NOT EXISTS ix_syscall_evil_event ON system_call_events (evil, event_id);

-- ── Model Runs (Experiment Tracking) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_runs (
    id                  INTEGER     PRIMARY KEY AUTOINCREMENT,
    run_name            VARCHAR(100) NOT NULL,
    model_type          VARCHAR(50),
    dataset_name        VARCHAR(100),
    algorithm           VARCHAR(50),
    hyperparameters     TEXT,                       -- JSON blob
    accuracy            REAL,
    roc_auc             REAL,
    f1_weighted         REAL,
    f1_macro            REAL,
    precision_score     REAL,
    recall_score        REAL,
    extra_metrics       TEXT,                       -- JSON blob
    model_artifact_path VARCHAR(500),
    preprocessor_path   VARCHAR(500),
    created_by          INTEGER     REFERENCES users(id),
    created_at          DATETIME    DEFAULT CURRENT_TIMESTAMP,
    duration_seconds    REAL,
    notes               TEXT
);

-- ── Predictions ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER  PRIMARY KEY AUTOINCREMENT,
    model_run_id        INTEGER  REFERENCES model_runs(id),
    event_source        VARCHAR(20),                -- network|syscall|synthetic
    source_record_id    INTEGER,
    predicted_label     INTEGER,                    -- 0=normal, 1=attack
    predicted_class     VARCHAR(50),
    confidence          REAL,
    true_label          INTEGER,
    predicted_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── Alerts ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id                  INTEGER     PRIMARY KEY AUTOINCREMENT,
    title               VARCHAR(200) NOT NULL,
    description         TEXT,
    severity            VARCHAR(20)  DEFAULT 'medium',  -- critical|high|medium|low|info
    status              VARCHAR(30)  DEFAULT 'open',    -- open|acknowledged|resolved|false_positive
    source_dataset      VARCHAR(50),
    attack_type         VARCHAR(50),
    confidence          REAL,
    model_run_id        INTEGER     REFERENCES model_runs(id),
    prediction_id       INTEGER     REFERENCES predictions(id),
    created_at          DATETIME    DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at     DATETIME,
    resolved_at         DATETIME,
    acknowledged_by     INTEGER     REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS ix_alerts_status     ON alerts (status);
CREATE INDEX IF NOT EXISTS ix_alerts_created_at ON alerts (created_at);

-- ── Analytics Views ───────────────────────────────────────────────────────────
-- Materialized as views for fast dashboard queries

CREATE VIEW IF NOT EXISTS v_attack_distribution AS
    SELECT
        attack_cat,
        COUNT(*)                                        AS total_events,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct_of_attacks
    FROM network_events
    WHERE label = 1
    GROUP BY attack_cat
    ORDER BY total_events DESC;

CREATE VIEW IF NOT EXISTS v_model_leaderboard AS
    SELECT
        run_name,
        algorithm,
        dataset_name,
        roc_auc,
        f1_weighted,
        accuracy,
        created_at
    FROM model_runs
    ORDER BY roc_auc DESC;

CREATE VIEW IF NOT EXISTS v_alert_summary AS
    SELECT
        severity,
        status,
        COUNT(*) AS alert_count
    FROM alerts
    GROUP BY severity, status
    ORDER BY
        CASE severity
            WHEN 'critical' THEN 1
            WHEN 'high'     THEN 2
            WHEN 'medium'   THEN 3
            WHEN 'low'      THEN 4
            ELSE 5
        END;
