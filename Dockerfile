# Build stage
FROM rust:1.87-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml ./

# Create dummy main.rs to cache dependencies
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source code
COPY src ./src

# Build for release
RUN touch src/main.rs && \
    cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/openai_responses_proxy /app/

# Create non-root user
RUN useradd -m -u 1000 appuser

# Create logs directory and set ownership
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Volume for logs
VOLUME ["/app/logs"]

# Note: Actual port is controlled by HOST_PORT env var (default: 8282)
EXPOSE 8282

CMD ["/app/openai_responses_proxy"]

