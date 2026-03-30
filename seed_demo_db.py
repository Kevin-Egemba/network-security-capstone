"""
Auto-seed script for Streamlit Cloud deployment.
Creates sample data so the dashboard has something to show.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.db.connector import DatabaseManager
from src.db.models import NetworkEvent, SystemCallEvent, ModelRun, Alert, AlertSeverity, AlertStatus, DatasetRegistry
from datetime import datetime
import random

def seed():
    mgr = DatabaseManager()
    mgr.create_tables()
    mgr.bootstrap_admin()

    with mgr.get_session() as session:
        # Only seed if empty
        if session.query(NetworkEvent).count() > 0:
            print("Database already seeded.")
            return

        print("Seeding database with sample data...")

        # Register dataset
        registry = DatasetRegistry(
            name="unsw_nb15_demo",
            source_file="demo",
            row_count=5000,
            column_count=45,
            description="UNSW-NB15 demo sample"
        )
        session.add(registry)
        session.flush()

        # Sample network events
        attack_cats = ["DoS", "Exploits", "Reconnaissance", "Generic", "Fuzzers", "Backdoor", "Analysis", "Shellcode", "Worms"]
        protos = ["tcp", "udp", "icmp", "arp"]
        states = ["FIN", "INT", "REQ", "CON", "RST"]

        events = []
        for i in range(5000):
            is_attack = random.random() > 0.55
            events.append(NetworkEvent(
                dataset_id=registry.id,
                split="train",
                proto=random.choice(protos),
                service=random.choice(["-", "http", "ftp", "smtp", "ssh", "dns"]),
                state=random.choice(states),
                dur=round(random.uniform(0, 100), 4),
                sbytes=random.randint(0, 100000),
                dbytes=random.randint(0, 100000),
                sttl=random.randint(0, 255),
                dttl=random.randint(0, 255),
                spkts=random.randint(1, 1000),
                dpkts=random.randint(1, 1000),
                label=1 if is_attack else 0,
                attack_cat=random.choice(attack_cats) if is_attack else "Normal",
            ))
        session.bulk_save_objects(events)

        # Sample syscall events
        syscalls = []
        for i in range(3000):
            syscalls.append(SystemCallEvent(
                dataset_id=registry.id,
                split="train",
                event_id=random.randint(1, 500),
                args_num=random.randint(0, 10),
                return_value=random.randint(-1, 100),
                sus=round(random.uniform(0, 1), 4),
                evil=1 if random.random() > 0.85 else 0,
            ))
        session.bulk_save_objects(syscalls)

        # Sample model runs
        algorithms = ["RandomForest", "XGBoost", "LogisticRegression", "IsolationForest", "KMeans"]
        datasets = ["UNSW-NB15", "BETH", "Cyber Attacks"]
        for i, algo in enumerate(algorithms):
            session.add(ModelRun(
                run_name=f"run_{algo.lower()}_{i+1}",
                model_type="supervised" if i < 3 else "unsupervised",
                dataset_name=random.choice(datasets),
                algorithm=algo,
                accuracy=round(random.uniform(0.80, 0.99), 4),
                roc_auc=round(random.uniform(0.85, 0.985), 4),
                f1_weighted=round(random.uniform(0.80, 0.98), 4),
                f1_macro=round(random.uniform(0.75, 0.97), 4),
            ))

        # Sample alerts
        alert_data = [
            ("High-volume DoS attack detected", AlertSeverity.critical, "DoS", 0.94),
            ("Exploit attempt on HTTP service", AlertSeverity.high, "Exploits", 0.88),
            ("Fuzzing activity on port 443", AlertSeverity.medium, "Fuzzers", 0.71),
            ("Reconnaissance scan detected", AlertSeverity.low, "Reconnaissance", 0.65),
            ("Backdoor connection attempt", AlertSeverity.critical, "Backdoor", 0.91),
        ]
        for title, severity, attack_type, confidence in alert_data:
            session.add(Alert(
                title=title,
                severity=severity,
                status=AlertStatus.open,
                attack_type=attack_type,
                confidence=confidence,
                source_dataset="unsw_nb15",
                description=f"{attack_type} detected with {confidence:.0%} confidence."
            ))

        session.commit()
        print("Database seeded successfully!")
        print(f"  Network events: 5,000")
        print(f"  Syscall events: 3,000")
        print(f"  Model runs: {len(algorithms)}")
        print(f"  Alerts: {len(alert_data)}")

if __name__ == "__main__":
    seed()
