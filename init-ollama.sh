#!/bin/bash
set -e

echo "Checking if llama3.1:8b model is available..."

# Check if model exists
if ollama list | grep -q "llama3.1:8b"; then
    echo "Model already exists, skipping download"
else
    echo "Model not found, pulling llama3.1:8b (this may take several minutes)..."
    ollama pull llama3.1:8b
    echo "Model download complete"
fi

# Keep the container running
exec ollama serve

