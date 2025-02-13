subset=$1
batch_size=32
sinkhorn_iteration=3
data_path="./data/QM9" 
weight_path="./checkpoints_QM9/checkpoint_best.pt"
results_path="./infer_results_QM9"

export CUDA_VISIBLE_DEVICES=0
python ./WGFormer/infer.py --user-dir ./WGFormer $data_path --valid-subset $subset \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task WGFormer --loss WGFormer --arch WGFormer \
       --path $weight_path \
       --num-sinkhorn-iteration $sinkhorn_iteration \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple 
