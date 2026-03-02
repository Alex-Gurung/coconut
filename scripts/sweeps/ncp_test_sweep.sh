# for CKPT in 2 3 8 9 11 12; do
for run in 0 1 2 3 4; do
# for run in 4; do
    # sed -i "s|load_model_path: \".*/checkpoint_[0-9]*\"|load_model_path: \"/mnt/disk/coconut/checkpoints/qwen-coconut-v4/checkpoint_${CKPT}\"|" args/qwen_coconut_ncpeval_v4.yml 
    # sed -i "s|load_model_path: \".*/checkpoint_[0-9]*\"|load_model_path: \"/mnt/disk/coconut/checkpoints/qwen-coconut-v4/checkpoint_14\"|" args/qwen_coconut_ncpeval_v4.yml 
    sed -i \
        -e "s|load_model_path: \".*/checkpoint_[0-9]*\"|load_model_path: \"/mnt/disk/coconut/checkpoints/qwen-coconut-v4/checkpoint_4\"|" \
        -e "s|resume: [0-9]*|resume: 4|" \
        args/qwen_coconut_ncpeval_v4.yml
   
    torchrun --nnodes 1 --nproc_per_node 4 run.py args/qwen_coconut_ncpeval_v4.yml 
    
    mv checkpoints/qwen-coconut-v4/eval_outputs.json \
       checkpoints/qwen-coconut-v4/check4_test${run}_eval_outputs.json
done

