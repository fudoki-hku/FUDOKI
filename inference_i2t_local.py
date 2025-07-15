import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.distributed as dist
from torchvision import transforms
import torch.backends.cudnn as cudnn

from fudoki.eval_loop import CFGScaledModel
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model


VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the script with custom arguments.")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--text_embedding_path", type=str, required=True, help="Path to the text embedding.")
    parser.add_argument("--image_embedding_path", type=str, required=True, help="Path to the image embedding.")
    parser.add_argument("--discrete_fm_steps", type=int, default=128, help="Inference steps for discrete flow matching")
    parser.add_argument("--txt_max_length", type=int, default=500, help="Text length maximum")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    return parser.parse_args()


def resize_pad(image, image_size=384):
    w, h = image.size
    if w <= 0 or h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    
    resize_scale = image_size / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))
    
    padding_color = (127, 127, 127)
    new_image = Image.new('RGB', (image_size, image_size), padding_color)
    
    if new_w <= 0 or new_h <= 0:
        return image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    paste_x = (image_size - new_w) // 2
    paste_y = (image_size - new_h) // 2
    
    new_image.paste(image, (paste_x, paste_y))
    return new_image


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
    image_path = args.image_path
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
        
    cfg_weighted_model = CFGScaledModel(model=model, g_or_u='understanding')
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

        img = Image.open(image_path).convert("RGB")

        generation_understanding_indicator = 0 # this is an understanding sample
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>What sports is shown in the image?"
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=vl_chat_processor.system_prompt,
        )
        if '<image_placeholder>' in sft_format:
            transform = transforms.Compose([
                transforms.Lambda(resize_pad),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            img = transform(img)
            img_len = IMG_LEN
        else:
            img = None
            img_len = IMG_LEN
        # tokenize
        input_ids = vl_chat_processor.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)
        # add image tokens to the input_ids
        image_token_mask = (input_ids == vl_chat_processor.image_id)
        image_indices = image_token_mask.nonzero()
        input_ids, _ = vl_chat_processor.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # pad tokens
        original_input_id_len = input_ids.shape[0]
    
        if original_input_id_len >= txt_max_length + img_len:
            raise ValueError("Sentences too long, not supported so far...")
        else:
            rows_to_pad = txt_max_length+img_len-input_ids.shape[0]
            input_ids = torch.cat([input_ids, torch.LongTensor([vl_chat_processor.pad_id]).repeat(rows_to_pad)], dim=0)
            attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
            attention_mask[:] = True
        
        # obtain image token mask and fill in img token_ids
        if img is not None:
            image_expanded_token_mask = (input_ids == vl_chat_processor.image_id).to(dtype=int)
            image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
            input_ids[image_expanded_mask_indices] = 0
        else:
            image_expanded_token_mask = torch.zeros_like(input_ids)
        
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
            text_expanded_token_mask[(start_index+split_token_length):] = 1
        else:
            raise ValueError("Split token not found in input_ids")

        generation_or_understanding_mask = generation_understanding_indicator
        data_info = dict()
        data_info['text_token_mask'] = text_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['image_token_mask'] = image_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['generation_or_understanding_mask'] = torch.Tensor([generation_or_understanding_mask]).unsqueeze(0).repeat(batch_size, 1).to(device).to(dtype=int)

        data_info['attention_mask'] = attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
        data_info['sft_format'] = sft_format
        if img is not None:
            data_info['understanding_img'] = img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)
        else:
            data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).repeat(batch_size, 1).to(device)
        

        x_0_txt = torch.randint(VOCABULARY_SIZE_TXT, input_ids.shape, dtype=torch.long, device=device)
        x_init = x_0_txt * data_info['text_token_mask'] + input_ids * (1 - data_info['text_token_mask'])

        synthetic_samples = solver.sample(
            x_init=x_init,
            step_size=1.0/discrete_fm_steps,
            verbose=True,
            return_intermediates=False,
            div_free=0,
            dtype_categorical=torch.float32,
            datainfo=data_info,
            cfg_scale=0,
        )
        
        sentences = vl_chat_processor.tokenizer.batch_decode(synthetic_samples, skip_special_tokens=True)
 
    for k, sentence in enumerate(sentences):   
        save_file_name = f"{save_path}/sample_{k}_understanding.txt"
        with open(save_file_name, "w") as file:
            file.write("response_sentences:"+ sentence + "\n"*10) 
        print(sft_format, sentence)
        print(f"Sentences have been written to {save_file_name}")


