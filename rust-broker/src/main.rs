use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgPoolOptions, Pool, Postgres, Row};
use std::{env, net::SocketAddr, sync::Arc};
use tokio::signal;
use tracing::{error, info};
use uuid::Uuid;

use lapin::{
    options::{BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Channel, Connection, ConnectionProperties,
};

#[derive(Clone)]
struct AppState {
    db: Pool<Postgres>,
    amqp: Arc<Channel>,
    queue_name: String,
}

#[derive(Deserialize)]
struct IrisReq {
    features: Vec<f64>, // [sepal_len, sepal_wid, petal_len, petal_wid]
}

#[derive(Deserialize)]
struct DiabetesFeatures {
    age: f64,
    sex: f64,
    bmi: f64,
    bp: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    s5: f64,
    s6: f64,
}
#[derive(Deserialize)]
struct DiabetesReq {
    features: DiabetesFeatures,
}

#[derive(Serialize)]
struct JobResp {
    id: Uuid,
    status: String,
}

#[derive(Serialize, Deserialize, sqlx::FromRow)]
struct JobRow {
    id: Uuid,
    model: String,
    status: String,
    input: serde_json::Value,
    result: Option<serde_json::Value>,
    error: Option<String>,
}

#[derive(Serialize)]
struct JobListItem {
    id: Uuid,
    model: String,
    status: String,
    created_at: String,
    updated_at: String,
    result: Option<serde_json::Value>,
    error: Option<String>,
}

#[derive(Debug, thiserror::Error)]
enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("not found")]
    NotFound,
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (code, msg) = match &self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, m.clone()),
            ApiError::NotFound => (StatusCode::NOT_FOUND, "not found".into()),
            ApiError::Internal(e) => {
                error!("internal error: {e:?}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error".into())
            }
        };
        (code, msg).into_response()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    dotenvy::dotenv().ok();

    // Optional: delay start so DB/MQ can come up in small clusters
    let startup_delay = env::var("STARTUP_DELAY_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5);
    if startup_delay > 0 {
        info!("Waiting {startup_delay}s for services to start…");
        tokio::time::sleep(std::time::Duration::from_secs(startup_delay)).await;
    }

    // Build connection strings from env (or sensible defaults for in-cluster)
    let db_url = env::var("DATABASE_URL").unwrap_or_else(|_| {
        let host = env::var("DB_HOST").unwrap_or_else(|_| "postgres".into());
        let port = env::var("DB_PORT").unwrap_or_else(|_| "5432".into());
        let name = env::var("DB_NAME").unwrap_or_else(|_| "demo".into());
        let user = env::var("DB_USER").unwrap_or_else(|_| "demo".into());
        let pass = env::var("DB_PASSWORD").unwrap_or_else(|_| "demo".into());
        format!("postgresql://{user}:{pass}@{host}:{port}/{name}")
    });

    let amqp_url = env::var("AMQP_URL").unwrap_or_else(|_| {
        let host = env::var("RABBITMQ_HOST").unwrap_or_else(|_| "rabbitmq".into());
        let port = env::var("RABBITMQ_PORT").unwrap_or_else(|_| "5672".into());
        let user = env::var("RABBITMQ_USER").unwrap_or_else(|_| "demo".into());
        let pass = env::var("RABBITMQ_PASSWORD").unwrap_or_else(|_| "demo".into());
        format!("amqp://{user}:{pass}@{host}:{port}/%2f")
    });

    let queue_name = env::var("AMQP_QUEUE").unwrap_or_else(|_| "inference_requests".into());
    let http_port: u16 = env::var("PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8080);

    // Connect to Postgres (with a few retries)
    info!("Connecting to Postgres: {}", db_url);
    let mut db_attempts = 0;
    let db = loop {
        db_attempts += 1;
        match PgPoolOptions::new()
            .max_connections(10)
            .acquire_timeout(std::time::Duration::from_secs(30))
            .connect(&db_url)
            .await
        {
            Ok(pool) => break pool,
            Err(e) if db_attempts < 5 => {
                error!("Postgres connect failed (attempt {}): {}", db_attempts, e);
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }
            Err(e) => return Err(e.into()),
        }
    };

    // Ensure table exists
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS jobs (
          id UUID PRIMARY KEY,
          model TEXT NOT NULL,
          input JSONB NOT NULL,
          status TEXT NOT NULL,
          result JSONB,
          error TEXT,
          created_at TIMESTAMPTZ DEFAULT now(),
          updated_at TIMESTAMPTZ DEFAULT now()
        );
        "#,
    )
    .execute(&db)
    .await?;

    // Connect to RabbitMQ (with a few retries)
    info!("Connecting to RabbitMQ: {}", amqp_url);
    let mut amqp_attempts = 0;
    let (amqp_conn, channel) = loop {
        amqp_attempts += 1;
        match Connection::connect(&amqp_url, ConnectionProperties::default()).await {
            Ok(conn) => match conn.create_channel().await {
                Ok(ch) => break (conn, ch),
                Err(e) if amqp_attempts < 5 => {
                    error!("Create AMQP channel failed (attempt {}): {}", amqp_attempts, e);
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            },
            Err(e) if amqp_attempts < 5 => {
                error!("RabbitMQ connect failed (attempt {}): {}", amqp_attempts, e);
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                continue;
            }
            Err(e) => return Err(e.into()),
        }
    };
    // declare queue (non-durable for demo)
    channel
        .queue_declare(
            &queue_name,
            QueueDeclareOptions {
                passive: false,
                durable: false,
                auto_delete: false,
                exclusive: false,
                nowait: false,
            },
            FieldTable::default(),
        )
        .await?;

    let shared = AppState {
        db,
        amqp: Arc::new(channel),
        queue_name,
    };

    let app = Router::new()
        .route("/", get(|| async { "OK" }))
        .route("/ui", get(ui_page))
        .route("/predict/iris", post(predict_iris))
        .route("/predict/diabetes", post(predict_diabetes))
        .route("/jobs", get(list_jobs))
        .route("/jobs/:id", get(get_job))
        .with_state(shared);

    let addr = SocketAddr::from(([0, 0, 0, 0], http_port));
    info!("Listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn predict_iris(
    State(st): State<AppState>,
    Json(req): Json<IrisReq>,
) -> Result<Json<JobResp>, ApiError> {
    if req.features.len() != 4 {
        return Err(ApiError::BadRequest(
            "features must be length 4: [sepal_len, sepal_wid, petal_len, petal_wid]".into(),
        ));
    }
    enqueue_job(st, "iris", serde_json::json!({ "features": req.features })).await
}

async fn predict_diabetes(
    State(st): State<AppState>,
    Json(req): Json<DiabetesReq>,
) -> Result<Json<JobResp>, ApiError> {
    let df = req.features;
    let payload = serde_json::json!({
        "features": {
            "age": df.age, "sex": df.sex, "bmi": df.bmi, "bp": df.bp,
            "s1": df.s1, "s2": df.s2, "s3": df.s3, "s4": df.s4, "s5": df.s5, "s6": df.s6
        }
    });
    enqueue_job(st, "diabetes", payload).await
}

async fn enqueue_job(
    st: AppState,
    model: &str,
    input: serde_json::Value,
) -> Result<Json<JobResp>, ApiError> {
    let id = Uuid::new_v4();
    sqlx::query("INSERT INTO jobs (id, model, input, status) VALUES ($1, $2, $3, 'queued')")
        .bind(id)
        .bind(model)
        .bind(&input)
        .execute(&st.db)
        .await
        .map_err(|e| ApiError::Internal(e.into()))?;

    let msg = serde_json::json!({"job_id": id, "model": model, "input": input});
    let payload = serde_json::to_vec(&msg).unwrap();
    st.amqp
        .basic_publish(
            "",
            &st.queue_name,
            BasicPublishOptions::default(),
            &payload,
            BasicProperties::default(),
        )
        .await
        .map_err(|e| ApiError::Internal(e.into()))?
        .await
        .map_err(|e| ApiError::Internal(e.into()))?;

    Ok(Json(JobResp {
        id,
        status: "queued".into(),
    }))
}

async fn get_job(
    State(st): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<JobRow>, ApiError> {
    let uuid = Uuid::parse_str(&id).map_err(|_| ApiError::BadRequest("invalid uuid".into()))?;
    let row = sqlx::query_as::<_, JobRow>(
        r#"SELECT id, model, status, input, result, error FROM jobs WHERE id = $1"#,
    )
    .bind(uuid)
    .fetch_optional(&st.db)
    .await
    .map_err(|e| ApiError::Internal(e.into()))?
    .ok_or(ApiError::NotFound)?;
    Ok(Json(row))
}

async fn list_jobs(State(st): State<AppState>) -> Result<Json<Vec<JobListItem>>, ApiError> {
    let rows = sqlx::query(
        r#"
        SELECT
          id, model, status,
          result, error,
          created_at::text AS created_at_str,
          updated_at::text AS updated_at_str
        FROM jobs
        ORDER BY created_at DESC
        LIMIT 20
        "#,
    )
    .fetch_all(&st.db)
    .await
    .map_err(|e| ApiError::Internal(e.into()))?;

    let items = rows
        .into_iter()
        .map(|r| JobListItem {
            id: r.get("id"),
            model: r.get("model"),
            status: r.get("status"),
            result: r.get::<Option<serde_json::Value>, _>("result"),
            error: r.get::<Option<String>, _>("error"),
            created_at: r
                .get::<Option<String>, _>("created_at_str")
                .unwrap_or_default(),
            updated_at: r
                .get::<Option<String>, _>("updated_at_str")
                .unwrap_or_default(),
        })
        .collect();
    Ok(Json(items))
}

async fn ui_page() -> Html<&'static str> {
    Html(UI_HTML)
}

const UI_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AI Queue Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
  h1 { margin-bottom: .5rem; }
  .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(320px,1fr)); }
  fieldset { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; }
  button { padding: .5rem .75rem; border-radius: 6px; border: 1px solid #ccc; cursor: pointer; }
  code { background: #f5f5f5; padding: .2rem .35rem; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; }
  th, td { border-bottom: 1px solid #eee; padding: .5rem; text-align: left; font-size: .95rem; vertical-align: top; }
  .ok { color: #0a7f2e; } .fail { color: #b40000; } .queued { color: #925a00; }
  .muted { color: #666; }
  .nowrap { white-space: nowrap; }
  .result { max-width: 420px; overflow-wrap: anywhere; }
  .panel { background:#fafafa; border:1px solid #eee; border-radius:8px; padding:1rem; }
</style>
</head>
<body>
<h1>AI Queue Demo</h1>
<p>Submit a request, then watch its status under <strong>Recent Jobs</strong>. Click a Job ID to view full details. Polling every 2s.</p>

<div class="grid">
  <fieldset>
    <legend><strong>Iris</strong> (classification)</legend>
    <p class="muted"><small>features: [sepal_len, sepal_wid, petal_len, petal_wid]</small></p>
    <form id="iris-form">
      <input type="number" step="0.01" name="f0" value="5.1" /> 
      <input type="number" step="0.01" name="f1" value="3.5" />
      <input type="number" step="0.01" name="f2" value="1.4" />
      <input type="number" step="0.01" name="f3" value="0.2" />
      <br/><br/>
      <button type="submit">Submit Iris</button>
      <button type="button" id="iris-rand">Random</button>
    </form>
    <pre id="iris-out"></pre>
  </fieldset>

  <fieldset>
    <legend><strong>Diabetes</strong> (regression)</legend>
    <p class="muted"><small>features object with keys age, sex, bmi, bp, s1..s6 (standardized)</small></p>
    <form id="diab-form">
      <div id="diab-fields"></div>
      <br/>
      <button type="submit">Submit Diabetes</button>
      <button type="button" id="diab-rand">Random</button>
    </form>
    <pre id="diab-out"></pre>
  </fieldset>
</div>

<h2>Recent Jobs</h2>
<table>
  <thead>
    <tr>
      <th>ID</th><th>Model</th><th>Status</th><th class="nowrap">Result (summary)</th><th>Updated</th>
    </tr>
  </thead>
  <tbody id="jobs"></tbody>
</table>

<h2>Selected Job</h2>
<div id="selected" class="panel">
  <p class="muted">None selected.</p>
</div>
<pre id="job-json" class="panel" style="display:none"></pre>

<script>
const qs = (s)=>document.querySelector(s);
const jobsTbody = qs('#jobs');
const selectedDiv = qs('#selected');
const jobJson = qs('#job-json');
let selectedId = null;

function ell(s){ return s.length>12 ? s.slice(0,8)+'…'+s.slice(-4) : s }
function badge(status){
  if(status==='completed') return '<span class="ok">completed</span>';
  if(status==='failed') return '<span class="fail">failed</span>';
  return '<span class="queued">'+status+'</span>';
}
function summarizeResult(r, err){
  if(err) return '<span class="fail">'+err+'</span>';
  if(!r) return '<span class="muted">—</span>';
  try{
    if(r.type==='classification' && r.label_name!==undefined){
      return 'class: <strong>' + r.label_name + '</strong> (id=' + (r.label_id ?? '?') + ')';
    }
    if(r.type==='regression' && r.prediction!==undefined){
      const p = Number(r.prediction);
      return 'ŷ = <strong>' + (isFinite(p)? p.toFixed(3): p) + '</strong>';
    }
  }catch(e){}
  let s = JSON.stringify(r);
  if(s.length>96) s = s.slice(0,93)+'…';
  return s;
}
async function listJobs(){
  try{
    const r = await fetch('/jobs');
    const data = await r.json();
    jobsTbody.innerHTML = data.map(j =>
      "<tr>" +
        "<td><a href='#' data-job-id='" + j.id + "'><code title='" + j.id + "'>" + ell(j.id) + "</code></a></td>" +
        "<td>" + j.model + "</td>" +
        "<td>" + badge(j.status) + "</td>" +
        "<td class='result'>" + summarizeResult(j.result, j.error) + "</td>" +
        "<td class='nowrap'><small>" + j.updated_at + "</small></td>" +
      "</tr>").join("");
    // bind click handlers
    document.querySelectorAll('[data-job-id]').forEach(a=>{
      a.addEventListener('click', ev=>{
        ev.preventDefault();
        selectedId = a.getAttribute('data-job-id');
        loadJob(selectedId);
      });
    });
  }catch(e){ console.log(e); }
}
async function loadJob(id){
  try{
    const r = await fetch('/jobs/'+id);
    const data = await r.json();
         selectedDiv.innerHTML = '<div><strong>Job:</strong> <code>' + data.id + '</code> • <strong>Model:</strong> ' + data.model + ' • <strong>Status:</strong> ' + data.status + '</div>';
    jobJson.style.display = 'block';
    jobJson.textContent = JSON.stringify(data, null, 2);
  }catch(e){
    selectedDiv.innerHTML = '<span class="fail">Failed to load job ' + id + '</span>';
    jobJson.style.display = 'none';
  }
}
setInterval(()=>{
  listJobs();
  if(selectedId) loadJob(selectedId);
}, 2000);
listJobs();

// Iris form
qs('#iris-form').addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const f = ev.target;
  const feats = [f.f0.value, f.f1.value, f.f2.value, f.f3.value].map(parseFloat);
  const r = await fetch('/predict/iris', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({features:feats})});
  qs('#iris-out').textContent = await r.text();
});
qs('#iris-rand').addEventListener('click', ()=>{
  const rand = ()=> (Math.random()*6).toFixed(2);
  ['f0','f1','f2','f3'].forEach((n)=>{ qs('#iris-form [name=' + n + ']').value = rand(); });
});

// Diabetes form (build inputs)
const diabKeys = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"];
qs('#diab-fields').innerHTML = diabKeys.map(k=>"<label>" + k + ": <input type='number' step='0.0001' name='" + k + "' value='0.0'/></label>").join("<br/>");
qs('#diab-form').addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  const f = ev.target;
  const features = Object.fromEntries(diabKeys.map(k=>[k, parseFloat(f[k].value)]));
  const r = await fetch('/predict/diabetes', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({features})});
  qs('#diab-out').textContent = await r.text();
});
qs('#diab-rand').addEventListener('click', ()=>{
  diabKeys.forEach(k => { qs('#diab-form [name=' + k + ']').value = (Math.random()*0.2 - 0.1).toFixed(4); });
});
</script>
</body>
</html>
"#;

async fn shutdown_signal() {
    let _ = signal::ctrl_c().await;
}
