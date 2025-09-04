use axum::{
    extract::{Path, State, Multipart},
    http::StatusCode,
    response::IntoResponse,
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
    options::{BasicPublishOptions, QueueDeclareOptions, QueuePurgeOptions},
    types::FieldTable,
    BasicProperties, Channel, Connection, ConnectionProperties,
};

use tower_http::services::ServeDir;

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

#[derive(Serialize)]
struct PurgeResp {
    deleted_jobs: i64,
    purged_messages: u32,
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

    // Ensure tables exist
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
        )
        "#,
    )
    .execute(&db)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS files (
          id UUID PRIMARY KEY,
          job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
          filename TEXT NOT NULL,
          content_type TEXT,
          file_size BIGINT NOT NULL,
          file_path TEXT NOT NULL,
          file_hash TEXT,
          created_at TIMESTAMPTZ DEFAULT now()
        )
        "#,
    )
    .execute(&db)
    .await?;

    // Connect to RabbitMQ (with a few retries)
    info!("Connecting to RabbitMQ: {}", amqp_url);
    let mut amqp_attempts = 0;
    let (_amqp_conn, channel) = loop {
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

    // --- Static UI serving ---
    // Keep assets under ./static/ui in the container (see Dockerfile COPY).
    let ui_dir = ServeDir::new("static/ui").append_index_html_on_directories(true);

    let app = Router::new()
        .route("/", get(|| async { "OK" }))
        // UI: serve static files at /ui/* and index at /ui
        .nest_service("/ui", ui_dir)
        // Also serve individual static files directly in case of routing issues
        .nest_service("/static", ServeDir::new("static"))
        // API
        .route("/predict/iris", post(predict_iris))
        .route("/predict/diabetes", post(predict_diabetes))
        .route("/predict/yolo", post(predict_yolo))
        .route("/predict/audio", post(predict_audio))
        .route("/jobs", get(list_jobs))
        .route("/jobs/:id", get(get_job))
        .route("/admin/purge", post(purge_all))
        .route("/debug/files", get(debug_files))
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

async fn predict_yolo(
    State(st): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<JobResp>, ApiError> {
    let mut image_data = None;
    let mut filename = None;
    
    // Process multipart form
    while let Some(field) = multipart.next_field().await
        .map_err(|e| ApiError::BadRequest(format!("Multipart error: {}", e)))? 
    {
        let name = field.name().unwrap_or("unknown").to_string();
        
        if name == "image" {
            filename = Some(field.file_name()
                .unwrap_or("image.jpg")
                .to_string());
            
            // Read file data directly into memory
            let data = field.bytes().await
                .map_err(|e| ApiError::BadRequest(format!("Failed to read image data: {}", e)))?;
            
            // Check file size limit (5MB for demo)
            const MAX_FILE_SIZE: usize = 5 * 1024 * 1024; // 5MB
            if data.len() > MAX_FILE_SIZE {
                return Err(ApiError::BadRequest(
                    format!("Image too large: {:.1}MB. Max size: 5MB", 
                           data.len() as f64 / 1024.0 / 1024.0)
                ));
            }
            
            // Encode as base64 for JSON transport
            image_data = Some(base64::encode(&data));
            break;
        }
    }
    
    let image_data = image_data.ok_or_else(|| ApiError::BadRequest("No image file provided".into()))?;
    let filename = filename.unwrap_or_else(|| "image.jpg".to_string());
    
    let payload = serde_json::json!({
        "image_data": image_data,
        "filename": filename
    });
    
    enqueue_job(st, "yolo", payload).await
}

async fn predict_audio(
    State(st): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<JobResp>, ApiError> {
    let mut audio_data = None;
    let mut filename = None;
    
    // Process multipart form
    while let Some(field) = multipart.next_field().await
        .map_err(|e| ApiError::BadRequest(format!("Multipart error: {}", e)))? 
    {
        let name = field.name().unwrap_or("unknown").to_string();
        
        if name == "audio" {
            filename = Some(field.file_name()
                .unwrap_or("audio.wav")
                .to_string());
            
            // Read file data directly into memory
            let data = field.bytes().await
                .map_err(|e| ApiError::BadRequest(format!("Failed to read audio data: {}", e)))?;
            
            // Check file size limit (10MB for audio)
            const MAX_AUDIO_SIZE: usize = 10 * 1024 * 1024; // 10MB
            if data.len() > MAX_AUDIO_SIZE {
                return Err(ApiError::BadRequest(
                    format!("Audio file too large: {:.1}MB. Max size: 10MB", 
                           data.len() as f64 / 1024.0 / 1024.0)
                ));
            }
            
            // Encode as base64 for JSON transport
            audio_data = Some(base64::encode(&data));
            break;
        }
    }
    
    let audio_data = audio_data.ok_or_else(|| ApiError::BadRequest("No audio file provided".into()))?;
    let filename = filename.unwrap_or_else(|| "audio.wav".to_string());
    
    let payload = serde_json::json!({
        "audio_data": audio_data,
        "filename": filename
    });
    
    enqueue_job(st, "audio", payload).await
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

async fn purge_all(State(st): State<AppState>) -> Result<Json<PurgeResp>, ApiError> {
    // count & truncate jobs
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM jobs")
        .fetch_one(&st.db)
        .await
        .map_err(|e| ApiError::Internal(e.into()))?;
    sqlx::query("TRUNCATE TABLE jobs")
        .execute(&st.db)
        .await
        .map_err(|e| ApiError::Internal(e.into()))?;

    // purge the queue
    let purged_messages = st
        .amqp
        .queue_purge(
            &st.queue_name,
            QueuePurgeOptions { nowait: false },
        )
        .await
        .map_err(|e| ApiError::Internal(e.into()))?;

    Ok(Json(PurgeResp {
        deleted_jobs: count,
        purged_messages,
    }))
}

async fn debug_files() -> Result<Json<serde_json::Value>, ApiError> {
    use std::fs;
    
    let cwd = std::env::current_dir().unwrap_or_else(|_| "unknown".into());
    let mut info = serde_json::json!({
        "working_directory": cwd,
        "files": {}
    });
    
    // Check if static directory exists
    let paths_to_check = [".", "static", "static/ui"];
    for path in paths_to_check {
        if let Ok(entries) = fs::read_dir(path) {
            let files: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            info["files"][path] = serde_json::json!(files);
        } else {
            info["files"][path] = serde_json::json!("directory_not_found");
        }
    }
    
    Ok(Json(info))
}

async fn shutdown_signal() {
    let _ = signal::ctrl_c().await;
}
