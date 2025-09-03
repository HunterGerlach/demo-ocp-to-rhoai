# --- build stage ---
    FROM rust:1.82 as builder
    WORKDIR /src
    
    # Needed for native-tls / openssl-sys during compile
    RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev ca-certificates && rm -rf /var/lib/apt/lists/*
    
    # Cache deps (first build will generate a lock file)
    COPY Cargo.toml .
    RUN mkdir -p src && echo "fn main() {}" > src/main.rs
    # Ensure base64ct is pinned in the lockfile too (belt-and-suspenders)
    RUN cargo update -p base64ct --precise 1.7.3 || true
    RUN cargo build --release
    
    # Real source - force rebuild by removing dummy first
    RUN rm src/main.rs
    COPY src/ src/
    # Force cargo to rebuild by touching the source
    RUN touch src/main.rs
    RUN cargo build --release
    # Verify we built the real application
    RUN ls -la target/release/queue-broker
    
    # --- runtime stage ---
    FROM debian:bookworm-slim
    # runtime OpenSSL is typically needed when linking with native-tls
    RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libpq5 \
        && rm -rf /var/lib/apt/lists/*
    ENV RUST_LOG=info
    WORKDIR /app
    COPY --from=builder /src/target/release/queue-broker /usr/local/bin/queue-broker
    # Verify the binary exists and has execute permissions
    RUN ls -la /usr/local/bin/queue-broker
    # Test the binary can at least start (will fail on missing env vars, but that's expected)
    RUN timeout 5s /usr/local/bin/queue-broker --help || true
    EXPOSE 8080
    CMD ["/usr/local/bin/queue-broker"]
    