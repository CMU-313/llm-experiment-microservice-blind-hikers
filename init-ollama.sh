#!/bin/bash
set -e

echo "Starting ollama..."
ollama serve &
OLLAMA_PID=$!

cleanup() {
    echo "Shutting down ollama (pid: $OLLAMA_PID)..."
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for ollama to become ready..."
ready=0
for i in $(seq 1 60); do
    if ollama list >/dev/null 2>&1; then
        ready=1
        break
    fi
    echo "  ollama not ready yet (attempt $i/60), retrying..."
    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "ERROR: ollama failed to start within the expected time." >&2
    exit 1
fi

echo "Checking if llama3.1:8b model is available..."
if ollama list | grep -q "llama3.1:8b"; then
    echo "Model already exists, skipping download."
else
    echo "Model not found, pulling llama3.1:8b (this may take several minutes)..."
    ollama pull llama3.1:8b
    echo "Model download complete."
fi

echo "Ollama is ready and serving."
wait "$OLLAMA_PID"

