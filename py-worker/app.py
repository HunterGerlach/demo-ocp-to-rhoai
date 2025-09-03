import json
import os
import sys
import time
import pickle
from typing import Dict, Any

import numpy as np
import pika
import psycopg2
from huggingface_hub import hf_hub_download

# ---- Config ----
AMQP_URL = os.getenv("AMQP_URL") or "amqp://{u}:{p}@{h}:{port}/%2f".format(
    u=os.getenv("RABBITMQ_USER", "demo"),
    p=os.getenv("RABBITMQ_PASSWORD", "demo"),
    h=os.getenv("RABBITMQ_HOST", "rabbitmq"),
    port=os.getenv("RABBITMQ_PORT", "5672"),
)
QUEUE_NAME = os.getenv("AMQP_QUEUE", "inference_requests")

DB_URL = os.getenv("DATABASE_URL") or "postgresql://{u}:{p}@{h}:{port}/{db}".format(
    u=os.getenv("DB_USER", "demo"),
    p=os.getenv("DB_PASSWORD", "demo"),
    h=os.getenv("DB_HOST", "postgres"),
    port=os.getenv("DB_PORT", "5432"),
    db=os.getenv("DB_NAME", "demo"),
)

# Hugging Face repos + files
IRIS_REPO = os.getenv(
    "MODEL_IRIS_REPO",
    "skops-tests/iris-sklearn-1.0-logistic_regression-without-config",
)
IRIS_FILE = os.getenv("MODEL_IRIS_FILE", "skops-ehiqc2lv.pkl")  # .pkl
DIAB_REPO = os.getenv(
    "MODEL_DIAB_REPO",
    "skops-tests/tabularregression-sklearn-latest-hist_gradient_boosting_regressor-with-config-pickle",
)
DIAB_FILE = os.getenv("MODEL_DIAB_FILE", "skops-xcxb87en.pkl")  # .pkl

HF_TOKEN = os.getenv("HF_TOKEN", None)
HF_HOME = os.getenv("HF_HOME", "/tmp/hf-cache")
os.makedirs(HF_HOME, exist_ok=True)

def load_pickle_from_hf(repo: str, filename: str):
    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=HF_TOKEN,
        cache_dir=HF_HOME,
        local_dir="/tmp/models",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    with open(local_path, "rb") as f:
        return pickle.load(f)

def connect_db():
    for i in range(30):
        try:
            conn = psycopg2.connect(DB_URL)
            conn.autocommit = True
            return conn
        except Exception as e:
            print(f"[worker] waiting for DB ({i})... {e}", flush=True)
            time.sleep(2)
    raise RuntimeError("could not connect to Postgres")

def connect_mq():
    params = pika.URLParameters(AMQP_URL)
    for i in range(30):
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=False)
            return connection, channel
        except Exception as e:
            print(f"[worker] waiting for MQ ({i})... {e}", flush=True)
            time.sleep(2)
    raise RuntimeError("could not connect to RabbitMQ")

def run_inference(models: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    model_key = job["model"]
    if model_key == "iris":
        feats = job["input"]["features"]
        X = np.array(feats, dtype=float).reshape(1, -1)
        clf = models["iris"]
        y = clf.predict(X)[0]
        # map to names for readability
        # 0=setosa, 1=versicolor, 2=virginica (typical sklearn iris labels)
        names = ["setosa", "versicolor", "virginica"]
        name = names[int(y)] if int(y) in (0, 1, 2) else f"class_{int(y)}"
        prob = None
        try:
            prob = clf.predict_proba(X)[0].tolist()
        except Exception:
            pass
        return {"type": "classification", "label_id": int(y), "label_name": name, "proba": prob}

    elif model_key == "diabetes":
        f = job["input"]["features"]
        cols = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
        X = np.array([[f[c] for c in cols]], dtype=float)
        reg = models["diabetes"]
        y = float(reg.predict(X)[0])
        return {"type": "regression", "prediction": y}

    else:
        raise ValueError(f"unknown model: {model_key}")

def main():
    print("[worker] downloading models from Hugging Face...", flush=True)
    iris_model = load_pickle_from_hf(IRIS_REPO, IRIS_FILE)
    diab_model = load_pickle_from_hf(DIAB_REPO, DIAB_FILE)
    models = {"iris": iris_model, "diabetes": diab_model}
    print("[worker] models ready.", flush=True)

    conn = connect_db()
    cur = conn.cursor()

    connection, channel = connect_mq()
    print("[worker] consuming queue:", QUEUE_NAME, flush=True)

    def on_message(ch, method, properties, body):
        try:
            job = json.loads(body.decode("utf-8"))
            job_id = job["job_id"]
            result = run_inference(models, job)
            # update DB
            cur.execute(
                "UPDATE jobs SET status=%s, result=%s, error=NULL, updated_at=now() WHERE id=%s",
                ("completed", json.dumps(result), job_id),
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[worker] job {job_id} done", flush=True)
        except Exception as e:
            print(f"[worker] error: {e}", flush=True)
            try:
                job_id = json.loads(body.decode("utf-8")).get("job_id")
                cur.execute(
                    "UPDATE jobs SET status=%s, error=%s, updated_at=now() WHERE id=%s",
                    ("failed", str(e), job_id),
                )
            except Exception as inner:
                print(f"[worker] failed to record error: {inner}", flush=True)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=on_message, auto_ack=False)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        try:
            channel.stop_consuming()
        except Exception:
            pass
    connection.close()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
