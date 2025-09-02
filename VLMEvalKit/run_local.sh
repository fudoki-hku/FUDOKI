CKPT_PATH=/path/to/FUDOKI

torchrun \
    --nproc_per_node 8 --master-port 12358 \
    run.py --data MME --model understanding \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 32 \
    --output_dir ./fudoki_output/understanding/ \
    --judge chatgpt-0125 \
    --judge-args '{"verbose": true}' \
    --txt_max_length 500 \
    --seed 99

torchrun \
    --nproc_per_node 8 --master-port 12358 \
    run.py --data POPE --model understanding \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 32 \
    --output_dir ./fudoki_output/understanding/ \
    --judge chatgpt-0125 \
    --judge-args '{"verbose": true}' \
    --txt_max_length 500 \
    --seed 99


torchrun \
    --nproc_per_node 8 --master-port 12358 \
    run.py --data MMBench_DEV_EN --model understanding \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 32 \
    --output_dir ./fudoki_output/understanding/ \
    --judge chatgpt-0125 \
    --judge-args '{"verbose": true}' \
    --txt_max_length 500 \
    --seed 99



torchrun \
    --nproc_per_node 8 --master-port 12358 \
    run.py --data SEEDBench_IMG --model understanding \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 32 \
    --output_dir ./fudoki_output/understanding/ \
    --judge chatgpt-0125 \
    --judge-args '{"verbose": true}' \
    --txt_max_length 500 \
    --seed 99



torchrun \
    --nproc_per_node 8 --master-port 12358 \
    run.py --data MMMU_DEV_VAL --model understanding \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 32 \
    --output_dir ./fudoki_output/understanding/ \
    --judge gpt-4-0125 \
    --judge-args '{"verbose": true}' \
    --txt_max_length 500 \
    --seed 99


torchrun \
    --nproc_per_node 8 --master-port 12358 \
    run.py --data MMVet --model understanding \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 100 \
    --output_dir ./fudoki_output/understanding/ \
    --judge-args '{"verbose": true}' \
    --txt_max_length 500 \
    --seed 99


torchrun \
    --nproc_per_node 8 --master-port 12358 \
    run.py --data GQA_TestDev_Balanced --model understanding \
    --checkpoint_path $CKPT_PATH \
    --text_embedding_path $CKPT_PATH/text_embedding.pt \
    --image_embedding_path  $CKPT_PATH/image_embedding.pt \
    --discrete_fm_steps 20 \
    --output_dir ./fudoki_output/understanding/ \
    --judge chatgpt-0125 \
    --judge-args '{"verbose": true}' \
    --txt_max_length 500 \
    --seed 99
