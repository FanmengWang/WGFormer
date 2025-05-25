data_path="./data/QM9"
save_dir="./checkpoints_QM9"
n_gpu=1
lr=2e-5
epoch=500
warmup=0.06
batch_size=32
update_freq=1
MASTER_PORT=8888
dict_name="dict.txt"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
        --user-dir ./WGFormer --train-subset train --valid-subset valid \
        --num-workers 8 --ddp-backend=c10d \
        --task WGFormer --loss WGFormer --arch WGFormer \
        --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
        --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size \
        --update-freq $update_freq --seed 1 \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --log-interval 1000 --log-format simple --tensorboard-logdir $save_dir/tsb \
        --validate-interval 1 --keep-last-epochs 10 \
        --keep-interval-updates 10 --best-checkpoint-metric loss --patience 20 --all-gather-list-size 10240000 \
        --save-dir $save_dir --tmp-save-dir $save_dir/tmp --tensorboard-logdir $save_dir/tsb \
        --find-unused-parameters
  
