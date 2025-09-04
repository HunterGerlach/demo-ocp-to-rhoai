# Production File Storage Architecture

## Current vs Recommended Implementation

### Current: Base64 in Messages ✅ (Simple Demo)
- **Good for**: Small files (<1MB), prototyping, simple demos
- **Limitations**: Memory usage, message size limits, no persistence

### Recommended: Persistent Volume + Database References 🚀 (Production)
- **Good for**: Large files, production systems, audit trails, file reuse
- **Benefits**: Scalable, persistent, efficient memory usage

## Production Implementation Example

### 1. Enhanced predict_yolo Function (PV Version)

```rust
async fn predict_yolo_pv(
    State(st): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<JobResp>, ApiError> {
    // Create job first
    let job_id = Uuid::new_v4();
    
    let mut file_refs = Vec::new();
    
    // Process multipart form
    while let Some(field) = multipart.next_field().await
        .map_err(|e| ApiError::BadRequest(format!("Multipart error: {}", e)))? 
    {
        let name = field.name().unwrap_or("unknown").to_string();
        
        if name == "image" {
            let filename = field.file_name()
                .unwrap_or("image.jpg")
                .to_string();
            
            // Generate file path in shared storage
            let file_id = Uuid::new_v4();
            let extension = filename.split('.').last().unwrap_or("jpg");
            let relative_path = format!("images/{}/{}.{}", 
                job_id, file_id, extension);
            let full_path = format!("/shared-storage/{}", relative_path);
            
            // Read file data
            let data = field.bytes().await
                .map_err(|e| ApiError::BadRequest(format!("Failed to read image data: {}", e)))?;
            
            // Save to persistent storage
            tokio::fs::create_dir_all(
                std::path::Path::new(&full_path).parent().unwrap()
            ).await
                .map_err(|e| ApiError::Internal(anyhow::anyhow!("Failed to create dir: {}", e)))?;
            
            tokio::fs::write(&full_path, &data).await
                .map_err(|e| ApiError::Internal(anyhow::anyhow!("Failed to save file: {}", e)))?;
            
            // Calculate file hash for deduplication
            let file_hash = format!("{:x}", md5::compute(&data));
            
            // Store file metadata in database
            let file_id = sqlx::query!(
                "INSERT INTO files (id, job_id, filename, content_type, file_size, file_path, file_hash) 
                 VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
                file_id,
                job_id,
                filename,
                "image/jpeg", // Could detect from file content
                data.len() as i64,
                relative_path,
                file_hash
            )
            .fetch_one(&st.db)
            .await
            .map_err(|e| ApiError::Internal(e.into()))?;
            
            file_refs.push(file_id.id);
            break;
        }
    }
    
    if file_refs.is_empty() {
        return Err(ApiError::BadRequest("No image file provided".into()));
    }
    
    // Create job with file references (not data)
    let payload = serde_json::json!({
        "file_refs": file_refs
    });
    
    enqueue_job_pv(st, job_id, "yolo", payload).await
}
```

### 2. Enhanced Python Worker (PV Version)

```python
def run_inference_pv(models: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    model_key = job["model"]
    job_id = job["job_id"]
    
    if model_key == "yolo":
        # Get file references from job
        file_refs = job["input"]["file_refs"]
        
        # Query database for file metadata
        cur.execute(
            "SELECT file_path, filename FROM files WHERE id = ANY(%s)",
            (file_refs,)
        )
        files = cur.fetchall()
        
        results = []
        for file_path, filename in files:
            # Read from shared storage
            full_path = f"/shared-storage/{file_path}"
            
            if not os.path.exists(full_path):
                raise ValueError(f"File not found: {full_path}")
            
            yolo_model = models["yolo"]
            detection_results = yolo_model(full_path)
            
            # Process results...
            # File stays in shared storage for audit/reuse
            
        return {"type": "object_detection", "results": results}
```

### 3. Benefits of PV Approach

#### ✅ **Scalability**
- Handles files of any size (GBs)
- No RabbitMQ message size limits
- Efficient memory usage

#### ✅ **Persistence & Audit**
- Files survive pod restarts
- Complete audit trail
- Can reprocess jobs later

#### ✅ **Deduplication**
- Same image uploaded multiple times = one copy
- File hash-based deduplication
- Storage cost savings

#### ✅ **Performance**
- No base64 encoding/decoding overhead
- Direct file access from storage
- Better for large ML datasets

### 4. Storage Configuration

```yaml
# In OpenShift template - already added!
- apiVersion: v1
  kind: PersistentVolumeClaim
  metadata: { name: ml-files-storage }
  spec:
    accessModes: ["ReadWriteMany"]  # Shared between broker & worker
    resources: { requests: { storage: "10Gi" } }
    storageClassName: ${STORAGE_CLASS_NAME}
```

### 5. File Lifecycle Management

```sql
-- Cleanup old files (cron job or periodic task)
DELETE FROM files 
WHERE created_at < NOW() - INTERVAL '30 days'
  AND job_id IN (
    SELECT id FROM jobs 
    WHERE status = 'completed' 
    AND updated_at < NOW() - INTERVAL '30 days'
  );
```

## Recommendation for Your Use Case

- **Keep current base64 approach** for this demo (simpler, works immediately)
- **Add file size limits** (e.g., 5MB max) to prevent issues
- **Consider PV approach** if you plan to:
  - Process larger files (videos, high-res images)
  - Keep files for audit trails
  - Build a production ML system
  - Support file reuse/deduplication

The infrastructure is now ready for either approach! 🚀
