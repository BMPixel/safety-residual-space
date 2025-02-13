#!/bin/bash
poetry shell

# Create output directory if it doesn't exist
mkdir -p data/jailbreaks

# Run each jailbreak generation script
for jailbreak in gcg gptfuzz pair flipattack codechameleon renellm simple strong_reject; do
    echo "Generating jailbreak samples using $jailbreak..."
    python src/jailbreaks/$jailbreak.py
done

echo "All jailbreak samples generated successfully!"
