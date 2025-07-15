import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.distributed as dist
from torchvision import transforms
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt

from fudoki.eval_loop import CFGScaledModel
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model


VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576
CFG_SCALE = 5.0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the script with custom arguments.")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--text_embedding_path", type=str, required=True, help="Path to the text embedding.")
    parser.add_argument("--image_embedding_path", type=str, required=True, help="Path to the image embedding.")
    parser.add_argument("--discrete_fm_steps", type=int, default=128, help="Inference steps for discrete flow matching")
    parser.add_argument("--txt_max_length", type=int, default=500, help="Text length maximum")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    print('world_size', world_size)
    print('local_rank', local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    device = 'cuda'

    checkpoint_path = Path(args.checkpoint_path)
    model_path = args.checkpoint_path
    model = instantiate_model(
       model_path
    ).to(device).to(torch.float32)
    model.train(False)
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    batch_size = args.batch_size
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    discrete_fm_steps = args.discrete_fm_steps
    txt_max_length = args.txt_max_length
        
    cfg_weighted_model = CFGScaledModel(model=model, g_or_u='generation')    
    with torch.no_grad():
        path_txt = MixtureDiscreteSoftmaxProbPath(mode='text', embedding_path=args.text_embedding_path)
        path_img = MixtureDiscreteSoftmaxProbPath(mode='image', embedding_path=args.image_embedding_path)
        solver = MixtureDiscreteSoftmaxEulerSolver(
            model=cfg_weighted_model,
            path_txt=path_txt,
            path_img=path_img,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
        )

        generation_understanding_indicator = 1
        img = None
        conversation = [
            {
                "role": "User",
                "content": "A rabbit wears a blue scarf."
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
              
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        sft_format = sft_format + vl_chat_processor.image_start_tag
        input_ids = vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)
        img_start = input_ids.shape[0]
        input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.image_id]*IMG_LEN), torch.LongTensor([vl_chat_processor.image_end_id])])
        img_end = input_ids.shape[0] - 1

        # pad tokens
        original_input_id_len = input_ids.shape[0]
    
        if original_input_id_len >= txt_max_length + IMG_LEN:
            raise ValueError("Sentences too long, not supported so far...")
        else:
            rows_to_pad = txt_max_length+IMG_LEN-input_ids.shape[0]
            input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
            attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
            attention_mask[:original_input_id_len] = True
        
        
        # obtain image token mask and fill in img token_ids
        image_expanded_token_mask = torch.zeros_like(input_ids)
        image_expanded_token_mask[img_start: img_end] = True

        # obtain text token mask
        # We assume that there is only one turn for assistant to respond
        text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)
        split_token = vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
        split_token_length = len(split_token)
        
        start_index = -1
        for j in range(len(input_ids) - split_token_length + 1):
            if input_ids[j:j + split_token_length].numpy().tolist() == split_token:
                start_index = j
                break

        if start_index != -1:
            text_expanded_token_mask[1: (start_index+split_token_length)] = 1
        else:
            raise ValueError("Split token not found in input_ids")

                
        generation_or_understanding_mask = generation_understanding_indicator
        data_info = dict()
        data_info['text_token_mask'] = text_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['image_token_mask'] = image_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['generation_or_understanding_mask'] = torch.Tensor([generation_or_understanding_mask]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)

        data_info['attention_mask'] = attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['sft_format'] = sft_format
        if generation_or_understanding_mask == 1:
            data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).repeat(batch_size).to(device)
        else:
            if img is not None:
                data_info['understanding_img'] = img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
                data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int).repeat(batch_size).to(device)
            else:
                data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
                data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).repeat(batch_size).to(device)

        input_ids = torch.LongTensor(input_ids).unsqueeze(0).repeat(batch_size, 1).to(device)
        

        x_0_img = torch.randint(16384, input_ids.shape, dtype=torch.long, device=device)
        x_init = x_0_img * data_info['image_token_mask'] + input_ids * (1 - data_info['image_token_mask'])
        
        synthetic_samples = solver.sample(
            x_init=x_init,
            step_size=1.0/discrete_fm_steps,
            verbose=True,
            return_intermediates=False,
            div_free=0,
            dtype_categorical=torch.float32,
            datainfo=data_info,
            cfg_scale=CFG_SCALE,
        )
           
        synthetic_samples = model.gen_vision_model.decode_code(synthetic_samples, [synthetic_samples.shape[0], 8, 24, 24]) # output value is between [-1, 1]
        synthetic_samples = (synthetic_samples + 1) / 2.0
        synthetic_samples = torch.clamp(synthetic_samples, min=0.0, max=1.0)
        image = (synthetic_samples.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        for k in range(synthetic_samples.shape[0]):
            Image.fromarray(image[k]).save(f"{save_path}/sample_{k}_generation.png")