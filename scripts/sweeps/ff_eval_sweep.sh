for CKPT in 6 7 8 9 11 12 13; do
# for CKPT in 13; do
    # sed -i "s|load_model_path: \".*/checkpoint_[0-9]*\"|load_model_path: \"/mnt/disk/coconut/checkpoints/qwen-coconut-v4/checkpoint_${CKPT}\"|" args/legacy/qwen_coconut_ncpeval_v4.yml 
    sed -i \
        -e "s|load_model_path: \".*/checkpoint_[0-9]*\"|load_model_path: \"/mnt/disk/coconut/checkpoints/qwen-coconut-v4/checkpoint_${CKPT}\"|" \
        -e "s|resume: [0-9]*|resume: ${CKPT}|" \
        args/legacy/qwen_coconut_ncpeval_v4.yml

    torchrun --nnodes 1 --nproc_per_node 4 run.py args/legacy/qwen_coconut_ncpeval_v4.yml 
    
    mv checkpoints/qwen-coconut-v4/eval_outputs.json \
       checkpoints/qwen-coconut-v4/check${CKPT}_eval_outputs.json
done
