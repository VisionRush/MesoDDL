base_dir="./output_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
./train.py \
    --model Mesorch_512 \
    --ckpt None \
    --conv_pretrain True \
    --seg_pretrain_path None\
    --world_size 8 \
    --find_unused_parameters \
    --batch_size 6 \
    --data_path ../training_set.json \
    --epochs 20 \
    --lr 1e-5 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --test_data_path "./val_dataset.json" \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --seed 42 \
    --test_period 1 \
    --num_workers 12 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log