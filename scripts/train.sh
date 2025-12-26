#!/bin/bash
# Train script wrapper

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

python src/cli/train.py "$@"
