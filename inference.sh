subset=$1
batch_size=32
data_path="./data/QM9" 
results_path="./infer_results_QM9/"
weight_path="./checkpoints_QM9/checkpoint_best.pt"

export CUDA_VISIBLE_DEVICES=0
python ./WGFormer/infer.py --user-dir ./WGFormer $data_path --valid-subset $subset \
       --path $weight_path --results-path $results_path --batch-size $batch_size \
       --task WGFormer --loss WGFormer --arch WGFormer \
       --num-workers 8 --ddp-backend=c10d \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple 
