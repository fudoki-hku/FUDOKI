torchrun --nproc_per_node 1 inference_i2t_local.py \
    --batch_size 1 \
    --checkpoint_path /cache/fudoki \
    --text_embedding_path /cache/text_embedding.pt \
    --image_embedding_path  /cache/image_embedding.pt \
    --image_path asset/0.jpg \
    --discrete_fm_steps 32 \
    --output_dir /cache/fudoki_output