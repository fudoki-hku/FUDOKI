CKPT_PATH=/mnt/a100/datasets1/wangjin/pretrained_model/FUDOKI

torchrun --nproc_per_node 1 inference_t2i_local.py \
    --batch_size 1 \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 32 \
    --output_dir ./fudoki_output