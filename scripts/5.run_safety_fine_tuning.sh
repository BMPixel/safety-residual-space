#!/bin/bash
poetry shell
# Create checkpoint directories
mkdir -p ckpts/ssft ckpts/dpo

# Run SSFT and DPO for different training sizes
for size in 10 20 40 80 160; do
    echo "Running fine-tuning with training size $size per type..."
    
    # Run SSFT
    echo "Running SSFT..."
    accelerate launch src/utils/safety_ssft.py \
        --train_size_per_type $size \
        --output_dir "ckpts/ssft/t${size}_n1"
    
    # Run DPO
    echo "Running DPO..."
    accelerate launch src/utils/safety_dpo.py \
        --train_size_per_type $size \
        --output_dir "ckpts/dpo/t${size}_n1"
done

echo "Safety fine-tuning completed!" 