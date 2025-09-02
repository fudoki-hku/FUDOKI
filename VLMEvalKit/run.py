import json

import torch
import torch.distributed as dist
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlmeval.config import supported_VLM
from vlmeval.dataset.video_dataset_config import supported_video_datasets
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer


from torchvision import transforms
import copy
import numpy as np
import torch.backends.cudnn as cudnn

from fudoki.eval_loop import CFGScaledModel
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from flow_matching.solver import MixtureDiscreteSoftmaxEulerSolver
from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model


VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def resize_pad(image):
    # 获取原始尺寸
    w, h = image.size
    
    # 检查输入图片是否有效
    if w <= 0 or h <= 0:
        # 如果图片无效，返回一个384x384的resize后的图片
        return image.resize((384, 384), Image.Resampling.BILINEAR)
    
    # 计算缩放比例
    resize_scale = 384 / max(w, h)
    new_w = max(1, int(w * resize_scale))
    new_h = max(1, int(h * resize_scale))
    
    # 创建384x384的灰色背景图
    padding_color = (127, 127, 127)
    new_image = Image.new('RGB', (384, 384), padding_color)
    
    # 将原图缩放并直接返回resize的结果(如果尺寸无效)
    if new_w <= 0 or new_h <= 0:
        return image.resize((384, 384), Image.Resampling.BILINEAR)
        
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # 计算粘贴位置,使图片居中
    paste_x = (384 - new_w) // 2
    paste_y = (384 - new_h) // 2
    
    # 将缩放后的图片粘贴到灰色背景上
    new_image.paste(image, (paste_x, paste_y))
    return new_image


image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']


def generate_npy_filename(name, max_length=255):
    # 分离路径和文件名
    file_name = '_'.join(name.rsplit('/', 1))
    
    # 提取文件名和扩展名
    base_name, ext = os.path.splitext(file_name)
    
    # 替换扩展名为 .npy
    new_file_name = base_name + '.npy'
    
    # 仅当文件名超过最大长度时才处理
    if len(new_file_name) > max_length:
        # 计算需要截取的长度
        truncate_length = max_length - len('.npy') - 8  # 保留 8 位哈希值
        if truncate_length > 0:
            # 截取文件名并加上哈希值
            hash_suffix = str(hash(new_file_name))[-8:]  # 使用哈希值作为后缀
            new_file_name = new_file_name[:truncate_length] + '_' + hash_suffix + '.npy'
        else:
            # 如果文件名实在太短，直接使用哈希值
            new_file_name = str(hash(new_file_name))[-max_length:] + '.npy'
    
    # 重新组合路径和文件名
    return new_file_name
        

def generate_inner(self, message, dataset=None, args=None, device='cuda', vq_model=None):
    conversation = self.prepare_inputs(message)
   
    self.path_txt = MixtureDiscreteSoftmaxProbPath(mode='text', embedding_path=args.text_embedding_path)
    self.path_img = MixtureDiscreteSoftmaxProbPath(mode='image', embedding_path=args.image_embedding_path)

    self.cfg_weighted_model = CFGScaledModel(model=self, g_or_u='understanding')

    solver = MixtureDiscreteSoftmaxEulerSolver(
        model=self.cfg_weighted_model,
        path_txt=self.path_txt,
        path_img=None,
        vocabulary_size_txt=VOCABULARY_SIZE_TXT,
        vocabulary_size_img=VOCABULARY_SIZE_IMG,
    )
  
    data_info = {}

    for message in conversation:
        if "images" not in message:
            continue
        image_path = message["images"][0]
        break
   
    save_image_path = image_path.replace("/root", "./fudoki_output/understanding")
    new_conversation = copy.deepcopy(conversation)
    for i, item in enumerate(new_conversation):
        item['from'] = item.pop('role')
        if item['from'] == 'User':
            item['from'] = 'human'
        elif item['from'] == 'Assistant':
            item['from'] = 'gpt'
        if 'images' in item:
            del item['images']
        new_item  = {k: item[k] for k in ['from', 'content'] if k in item}
        new_conversation[i] = new_item

    
    self.conversation_list.append({"image": save_image_path, "conversations": new_conversation})
   
    image = Image.open(image_path).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Lambda(resize_pad),  # 自定义转换函数
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    image = transform(image).unsqueeze(0).to(device)
  
    img = image.squeeze(0)

    for message in conversation:
        if message["role"] == "User":
            query = message["content"]
            break
    
    for message in conversation:
        if message["role"] == "Assistant":
            ground_truth = message["content"]
            message["content"] = ""
            break
            
    sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt=self.vl_chat_processor.system_prompt,
        )
    
    query = sft_format
    print("query", query)

    # tokenize
    input_ids = self.vl_chat_processor.tokenizer.encode(query)
    input_ids = torch.LongTensor(input_ids)

    # add image tokens to the input_ids
    image_token_mask = (input_ids == self.vl_chat_processor.image_id)
    image_indices = image_token_mask.nonzero()
    input_ids, _ = self.vl_chat_processor.add_image_token(
        image_indices=image_indices,
        input_ids=input_ids,
    )

    # pad tokens
    original_input_id_len = input_ids.shape[0]
    txt_max_length = args.txt_max_length
    answer_token_num = args.txt_max_length
    img_len = IMG_LEN

    if original_input_id_len < txt_max_length + img_len:
        input_ids = torch.cat([input_ids, torch.LongTensor([self.vl_chat_processor.pad_id]).repeat(txt_max_length+img_len-input_ids.shape[0])], dim=0)
        attention_mask = torch.zeros((input_ids.shape[0]), dtype=torch.bool)
        attention_mask[:] = True
        attention_mask[original_input_id_len + answer_token_num:] = False
    else:
        return "Sentences too long, abandoning..."
  
    image_expanded_token_mask = (input_ids == self.vl_chat_processor.image_id).to(dtype=int)
    image_expanded_mask_indices = torch.where(image_expanded_token_mask == 1)[0]
   
    
    # obtain text token mask
    # We assume that there is only one turn for assistant to respond
    text_expanded_token_mask = torch.zeros_like(image_expanded_token_mask)

    split_token = self.vl_chat_processor.tokenizer.encode("Assistant:", add_special_tokens=False)
    split_token_length = len(split_token)
    
    start_index = -1
    for j in range(len(input_ids) - split_token_length + 1):
        if input_ids[j:j + split_token_length].numpy().tolist() == split_token:
            start_index = j
            break

    if start_index != -1:
        text_expanded_token_mask[(start_index+split_token_length):(original_input_id_len+answer_token_num)] = 1
    else:
        raise ValueError("Split token not found in input_ids")

    # print("text_expanded_token_mask shape", text_expanded_token_mask.shape)

    generation_or_understanding_mask = 0
    batch_size = 1
    data_info['text_token_mask'] = text_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
    data_info['image_token_mask'] = image_expanded_token_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
    data_info['generation_or_understanding_mask'] = torch.Tensor([generation_or_understanding_mask]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)

    data_info['attention_mask'] = attention_mask.unsqueeze(0).repeat(batch_size, 1).to(device)
    if generation_or_understanding_mask == 1:
        data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)
    else:
        if img is not None:
            data_info['understanding_img'] = img.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            data_info['has_understanding_img'] = torch.Tensor([True]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)
        else:
            data_info['understanding_img'] = torch.zeros((3, 384, 384)).unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            data_info['has_understanding_img'] = torch.Tensor([False]).to(dtype=int).unsqueeze(0).repeat(batch_size, 1).to(device)

    # print("attention_mask", attention_mask)

    input_ids = torch.LongTensor(input_ids).unsqueeze(0).repeat(batch_size, 1).to(device)
    

    x_0_txt = torch.randint(VOCABULARY_SIZE_TXT, input_ids.shape, dtype=torch.long, device=device)
    x_init = x_0_txt * data_info['text_token_mask'] + input_ids * (1 - data_info['text_token_mask'])
    
    steps = args.discrete_fm_steps

    if True:
    
        synthetic_samples = solver.sample(
            x_init=x_init,
            step_size=1.0/steps,
            verbose=True,
            div_free=0.0,
            dtype_categorical=torch.float32,
            datainfo=data_info,
            cfg_scale=0
        )
        
        sentences = self.vl_chat_processor.tokenizer.batch_decode(
            synthetic_samples[..., :answer_token_num].reshape(batch_size, -1)
        )
    
        def keep_strictly_before_eos(text, eos_token="<｜end▁of▁sentence｜>"):
            idx = text.find(eos_token)
            if idx == -1:
                return text  # eos_token not found, return the whole string
            return text[:idx]
        print(f"Original response: {sentences[0]}")
        response = keep_strictly_before_eos(sentences[0])
    
    else:
        response = ground_truth

    print("response::", response)
    answer = response
    return answer


def prepare_inputs(self, message):
    def prepare_itlist(msgs):
        content, images = '', []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
                content += '<image_placeholder>'
            elif s['type'] == 'text':
                content += s['value']
            
        return content, images
    conversation = []
    if 'role' not in message[0]:
        content, images = prepare_itlist(message)
        conversation.append(dict(role='User', content=content, images=images))
    else:
        role_map = {'user': 'User', 'assistant': 'Assistant'}
        for msgs in message:
            role = role_map[msgs['role']]
            content, images = prepare_itlist(msgs['content'])
            conversation.append(dict(role=role, content=content, images=images))
    for msgs in message:
        if msgs["type"] == "ground_truth":
            conversation.append(dict(role='Assistant', content=msgs['value']))
            break
    return conversation


def build_model_from_config(cfg, model_name):
    import vlmeval.api
    import vlmeval.vlm
    config = cp.deepcopy(cfg[model_name])
    if config == {}:
        return supported_VLM[model_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.api, cls_name):
        return getattr(vlmeval.api, cls_name)(**config)
    elif hasattr(vlmeval.vlm, cls_name):
        return getattr(vlmeval.vlm, cls_name)(**config)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.api` or `vlmeval.vlm`')


def build_dataset_from_config(cfg, dataset_name):
    import vlmeval.dataset
    import inspect
    config = cp.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    assert 'class' in config
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')


def parse_args():
    help_msg = """\
You can launch the evaluation by setting either --data and --model or --config.

--data and --model:
    Each Arg should be a list of strings, specifying the names of datasets and models.
    To find all supported model names, please refer to the `vlmeval/config.py` of check the output of the command \
        `vlmutil mlist all` in the terminal (you should first have vlmeval installed).
    To find all supported dataset names, please refer to the `vlmeval/dataset/__init__.py` file. The python script \
        to print all supported dataset names is as follows:
        ```python
        from vlmeval.dataset import SUPPORTED_DATASETS
        print(SUPPORTED_DATASETS)
        ```
        or you can check the output of the command `vlmutil dlist all` in the terminal.
    To find all supported video dataset default settings, please refer to the \
        `vlmeval/dataset/video_dataset_config.py` file.

--config:
    Launch the evaluation by specifying the path to the config json file. Sample Json Content:
    ```json
    {
        "model": {
            "GPT4o_20240806_T00_HIGH": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 0,
                "img_detail": "high"
            },
            "GPT4o_20240806_T10_Low": {
                "class": "GPT4V",
                "model": "gpt-4o-2024-08-06",
                "temperature": 1.0,
                "img_detail": "low"
            },
            "GPT4o_20241120": {}
        },
        "data": {
            "MME-RealWorld-Lite": {
                "class": "MMERealWorld",
                "dataset": "MME-RealWorld-Lite"
            },
            "MMBench_DEV_EN_V11": {
                "class": "ImageMCQDataset",
                "dataset": "MMBench_DEV_EN_V11"
            },
            "MMBench_Video_8frame_nopack": {},
            "Video-MME_16frame_subs": {
                "class": "VideoMME",
                "dataset": "Video-MME",
                "nframe": 16,
                "use_subtitle": true,
            }
        }
    }
    ```
    Currently, only `model` and `data` are supported fields. The content of each field is a dictionary.
    For `model`, the key is the name of the model, and the value is a dictionary containing the following keys:
    - `class`: The class name of the model, which should be a class in `vlmeval.vlm` or `vlmeval.api`.
    - Other keys are specific to the model, please refer to the corresponding class.
    - Tip: The defined model in the `supported_VLM` of `vlmeval/config.py` can be used as a shortcut.
    For `data`, the key is the name of the dataset (should be the same as the `dataset` field in most cases, \
        except for video datasets), and the value is a dictionary containing the following keys:
    - `class`: The class name of the dataset, which should be a class in `vlmeval.dataset`.
    - `dataset`: The name of the dataset, which should be a string that is accepted by the `dataset` argument of the \
        corresponding class.
    - Other keys are specific to the dataset, please refer to the corresponding class.
    - Tip: The defined dataset in the `supported_video_datasets` of `vlmeval/dataset/video_dataset_config.py` \
        can be used as a shortcut.

    The keys in the `model` and `data` fields will be used for naming the prediction files and evaluation results.
    When launching with `--config`, args for API VLMs, such as `--retry`, `--verbose`, will be ignored.
"""
    parser = argparse.ArgumentParser(description=help_msg, formatter_class=argparse.RawTextHelpFormatter)
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('--data', type=str, nargs='+', help='Names of Datasets')
    parser.add_argument('--model', type=str, nargs='+', help='Names of Models')
    parser.add_argument('--config', type=str, help='Path to the Config Json File')
    # Work Dir
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--api-nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    parser.add_argument('--judge-args', type=str, default=None, help='Judge arguments in JSON format')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Reuse: will reuse the existing prediction files
    parser.add_argument('--reuse', action='store_true')
    # Reuse-aux: if set, when reuse is True, will also reuse the auxiliary evaluation files
    parser.add_argument('--reuse-aux', type=bool, default=True, help='reuse auxiliary evaluation files')
   

    parser.add_argument("--seed", default=99, type=int)
    parser.add_argument("--discrete_fm_steps", type=int, default=128, help="Inference steps for discrete flow matching")
    parser.add_argument("--txt_max_length", type=int, default=500, help="Text length maximum")
    parser.add_argument("--text_embedding_path", type=str, required=True, help="Path to the text embedding.")
    parser.add_argument("--image_embedding_path", type=str, required=True, help="Path to the image embedding.")


    args = parser.parse_args()
    return args





def main():
    logger = get_logger('RUN')
    rank, world_size = get_rank_and_world_size()
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    use_config, cfg = False, None

    if args.config is not None:
        assert args.data is None and args.model is None, '--data and --model should not be set when using --config'
        use_config, cfg = True, load(args.config)
        args.model = list(cfg['model'].keys())
        args.data = list(cfg['data'].keys())
    else:
        assert len(args.data), '--data should be a list of data files'

    if rank == 0:
        if not args.reuse:
            logger.warning('--reuse is not set, will not reuse previous (before one day) temporary files')
        else:
            logger.warning('--reuse is set, will reuse the latest prediction & temporary pickle files')

    if 'MMEVAL_ROOT' in os.environ:
        args.work_dir = os.environ['MMEVAL_ROOT']

    if not use_config:
        for k, v in supported_VLM.items():
            if hasattr(v, 'keywords') and 'retry' in v.keywords and args.retry is not None:
                v.keywords['retry'] = args.retry
                supported_VLM[k] = v
            if hasattr(v, 'keywords') and 'verbose' in v.keywords and args.verbose is not None:
                v.keywords['verbose'] = args.verbose
                supported_VLM[k] = v

   
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(int(local_rank))
    dist.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 7200)))
    )
    device = torch.device(f'cuda:{local_rank}')

    save_path = args.output_dir
    model_path = args.checkpoint_path
    
    if args.data[0] in ["POPE", "MME"]:
        res_file = f"{save_path}/{args.model[0]}/understanding_{args.data[0]}_score.csv"
    elif args.data[0] in ["MMMU_DEV_VAL", "SEEDBench_IMG", "MMBench_DEV_EN", "GQA_TestDev_Balanced"]:
        res_file = f"{save_path}/{args.model[0]}/understanding_{args.data[0]}_acc.csv"
    else:
        res_file = f"{save_path}/{args.model[0]}/understanding_{args.data[0]}_gpt-4-turbo_score.csv"
   

    for _, model_name in enumerate(args.model):
        full_model_name = f"{model_name}"
        model = None
        date, commit_id = timestr('day'), githash(digits=8)
        eval_id = f"T{date}_G{commit_id}"

        pred_root = osp.join(args.output_dir, full_model_name, eval_id)
        pred_root_meta = osp.join(args.output_dir, full_model_name)
        os.makedirs(pred_root_meta, exist_ok=True)

        prev_pred_roots = ls(osp.join(args.output_dir, full_model_name), mode='dir')
        if len(prev_pred_roots) and args.reuse:
            prev_pred_roots.sort()

        if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

        
        # Load our model
        model = instantiate_model(
            model_path
        ).to(device).to(torch.float32)
        model.training = False
        model.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        print("Model has been successfully loaded.")

        import types
        import functools

        model.prepare_inputs = types.MethodType(prepare_inputs, model)
        generate_with_args = functools.partial(generate_inner, args=args, device=device)
        model.generate = types.MethodType(generate_with_args, model)
        
        print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

        model.conversation_list = []

        for _, dataset_name in enumerate(args.data):
            if world_size > 1:
                dist.barrier()

            try:
                result_file_base = f'{model_name}_{dataset_name}.xlsx'

                if use_config:
                    assert False
                    if world_size > 1:
                        if rank == 0:
                            dataset = build_dataset_from_config(cfg['data'], dataset_name)
                        dist.barrier()
                    dataset = build_dataset_from_config(cfg['data'], dataset_name)
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                else:
                    dataset_kwargs = {}
                    if dataset_name in ['MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI']:
                        dataset_kwargs['model'] = model_name

                    
                    if world_size > 1:
                        if rank == 0:
                            dataset = build_dataset(dataset_name, **dataset_kwargs)
                        dist.barrier()

                    dataset = build_dataset(dataset_name, **dataset_kwargs)
                    
                    if dataset is None:
                        logger.error(f'Dataset {dataset_name} is not valid, will be skipped. ')
                        continue
                   


                dataset.data = dataset.data
                if dataset.TYPE == 'MT':
                    result_file_base = result_file_base.replace('.xlsx', '.tsv')

                result_file = osp.join(pred_root, result_file_base)

                
                if rank == 0 and len(prev_pred_roots):
                    prev_result_files = []
                    prev_pkl_file_list = []
                    for root in prev_pred_roots[::-1]:
                        if osp.exists(osp.join(root, result_file_base)):
                            if args.reuse_aux:
                                prev_result_files = fetch_aux_files(osp.join(root, result_file_base))
                            else:
                                prev_result_files = [osp.join(root, result_file_base)]
                            break
                        elif commit_id in root and len(ls(root)) and root != pred_root:
                            temp_files = ls(root, match=[dataset_name, '.pkl'])
                            if len(temp_files):
                                prev_pkl_file_list.extend(temp_files)
                                break
                    if not args.reuse:
                        prev_result_files = []
                        prev_pkl_file_list = []
                    if len(prev_result_files):
                        for prev_result_file in prev_result_files:
                            src = prev_result_file
                            tgt = osp.join(pred_root, osp.basename(src))
                            if not osp.exists(tgt):
                                shutil.copy(src, tgt)
                                logger.info(f'--reuse is set, will reuse the prediction file {src}.')
                            else:
                                logger.warning(f'File already exists: {tgt}')
                        
                    elif len(prev_pkl_file_list):
                        for fname in prev_pkl_file_list:
                            target_path = osp.join(pred_root, osp.basename(fname))
                            if not osp.exists(target_path):
                                shutil.copy(fname, target_path)
                                logger.info(f'--reuse is set, will reuse the prediction pickle file {fname}.')
                            else:
                                logger.warning(f'File already exists: {target_path}')

                if world_size > 1:
                    dist.barrier()

                if model is None:
                    model = model_name  # which is only a name

                print("dataset.MODALITY", dataset.MODALITY)
                print("dataset name", dataset_name)
                # Perform the Inference
                if dataset.MODALITY == 'VIDEO':
                    model = infer_data_job_video(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        result_file_name=result_file_base,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc)
                elif dataset.TYPE == 'MT':
                    model = infer_data_job_mt(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        ignore_failed=args.ignore)
                else:
                    # pass
                    model = infer_data_job(
                        model,
                        work_dir=pred_root,
                        model_name=model_name,
                        dataset=dataset,
                        verbose=args.verbose,
                        api_nproc=args.api_nproc,
                        ignore_failed=args.ignore)

                # Set the judge kwargs first before evaluation or dumping

                judge_kwargs = {
                    'nproc': args.api_nproc,
                    'verbose': args.verbose,
                    'retry': args.retry if args.retry is not None else 3,
                    **(json.loads(args.judge_args) if args.judge_args else {}),
                }

                if args.retry is not None:
                    judge_kwargs['retry'] = args.retry
                if args.judge is not None:
                    judge_kwargs['model'] = args.judge
                else:
                    if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro']:
                        if listinstr(['WeMath'], dataset_name):
                            judge_kwargs['model'] = 'gpt-4o-mini'
                        else:
                            judge_kwargs['model'] = 'chatgpt-0125'
                    elif listinstr(['MMVet', 'LLaVABench', 'MMBench-Video'], dataset_name):
                        judge_kwargs['model'] = 'gpt-4-turbo'
                    elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o-mini'
                    elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision'], dataset_name):  # noqa: E501
                        judge_kwargs['model'] = 'gpt-4o'

                if rank == 0:
                    logger.info(judge_kwargs)

                if world_size > 1:
                    dist.barrier()

                # Only Rank 0 handles the evaluation part
                if rank == 0:
                    # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
                    if dataset_name in ['MMMU_TEST']:
                        result_json = MMMU_result_transfer(result_file)
                        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, '
                                    f'json file saved in {result_json}')
                        continue
                    elif 'MMT-Bench_ALL' in dataset_name:
                        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
                        logger.info(f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
                                    f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
                                    f'submission file saved in {submission_file}')
                        continue

                    # Skip the evaluation part if only infer
                    if args.mode == 'infer':
                        continue

                    # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
                    if 'MLLMGuard_DS' in dataset_name:
                        logger.info('The evaluation of MLLMGuard_DS is not supported yet. ')
                        continue
                    elif 'AesBench_TEST' == dataset_name:
                        logger.info(f'The results are saved in {result_file}. '
                                    f'Please send it to the AesBench Team via huangyipo@hotmail.com.')
                        continue
                    elif dataset_name in ['DocVQA_TEST', 'InfoVQA_TEST', 'Q-Bench1_TEST', 'A-Bench_TEST']:
                        logger.info(f'{dataset_name} is a test split without ground-truth. '
                                    'Thus only the inference part is supported for those datasets. ')
                        continue
                    elif dataset_name in [
                        'MMBench_TEST_CN', 'MMBench_TEST_EN', 'MMBench', 'MMBench_CN',
                        'MMBench_TEST_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_V11', 'MMBench_CN_V11'
                    ] and not MMBenchOfficialServer(dataset_name):
                        logger.error(
                            f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')
                        continue

                    # Setup the proxy for the evaluation
                    eval_proxy = os.environ.get('EVAL_PROXY', None)
                    old_proxy = os.environ.get('HTTP_PROXY', '')
                    if eval_proxy is not None:
                        proxy_set(eval_proxy)

                    print("start evaluation")
                    print("result_file", result_file)
                    # Perform the Evaluation
                    eval_results = dataset.evaluate(result_file, **judge_kwargs)

                    print("end evaluation")
                    # Display Evaluation Results in Terminal
                    if eval_results is not None:
                        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
                        logger.info(f'The evaluation of model {model_name} x dataset {dataset_name} has finished! ')
                        logger.info('Evaluation Results:')
                        if isinstance(eval_results, dict):
                            logger.info('\n' + json.dumps(eval_results, indent=4))
                        elif isinstance(eval_results, pd.DataFrame):
                            if len(eval_results) < len(eval_results.columns):
                                eval_results = eval_results.T
                            logger.info('\n' + tabulate(eval_results))

                    # Restore the proxy
                    if eval_proxy is not None:
                        proxy_set(old_proxy)

                    # Create the symbolic links for the prediction files
                    files = os.listdir(pred_root)
                    files = [x for x in files if (f'{model_name}_{dataset_name}' in x or "status.json" in x)]
                    for f in files:
                        cwd = os.getcwd()
                        file_addr = osp.join(cwd, pred_root, f)
                        link_addr = osp.join(cwd, pred_root_meta, f)
                        if osp.exists(link_addr) or osp.islink(link_addr):
                            os.remove(link_addr)
                        os.symlink(file_addr, link_addr)

            except Exception as e:
                logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                                'skipping this combination.')
                continue

            if world_size > 1:
                dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()
    print("all testing finished.")


if __name__ == '__main__':
    load_env()
    main()