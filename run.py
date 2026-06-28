import os
import time

import torch
import torch.nn as nn

from bigptq_arb import BRAGPTQ
from binary_arb import Binarization
from modelutils import find_layers
from tqdm import tqdm
import logging
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    HAS_QWEN_VL = True
except ImportError:
    HAS_QWEN_VL = False
import datetime
from my_utils import build_cu_seqlens_from_grid_thw
from utils.autosearch_arb import high_order_residual_alternating_order2_rc_nomean, high_order_residual_alternating_order1_rc_nomean
from datasets import load_dataset
from mllm_utils import (
    get_layer_hessian_sensitivity_multi,
    get_layer_hessian_sensitivity_multi_robust,
    modality_structural_partition,
    build_text_hessian_inputs,
    build_coco_vision_inputs,
    build_coco_inputs,
    build_mixed_inputs,
    quantize_4bit_bnb,
    build_mixed_inputs_fixed_window
)
from memory_recoder import trace_cuda_mem
from compute_method.gptq import GPTQ
from compute_method.quant import Quantizer
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    processor = None
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto", token=hf_token)
        model.seqlen = 2048
    elif "Qwen3" in model:
        model_name = model
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model.seqlen = 4096
    elif "Qwen2.5" in model:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_name = model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model.seqlen = 4096
    elif "InternVL3" in model:
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        model_name = model
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype="auto",
        )
        processor = AutoProcessor.from_pretrained(model_name)
        model.seqlen = 4096
    logging.info(f"{model}")
    logging.info(f"seqlen: {model.seqlen}")
    return model, processor

@torch.no_grad()
def quant_sequential(model, dataloader, dev,
                     sens_inputs_text_list=None,
                     sens_inputs_vl_list=None,
                     save_folder=None):
    model = model.to(dev)

    for name, module in model.named_modules(): 
        module.global_name = args.model + name

    if "Qwen" in model.__class__.__name__ or "InternVL" in model.__class__.__name__:
        use_cache = model.config.text_config.use_cache
        model.config.text_config.use_cache = False
    else:
        use_cache = model.config.use_cache
        model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    elif "Qwen" in args.model:
        layers = model.model.base_model.base_model.base_model.language_model.layers
        model.model.base_model.base_model.base_model.language_model.embed_tokens = model.model.base_model.base_model.base_model.language_model.embed_tokens.to(dev)
        model.model.base_model.base_model.base_model.language_model.norm = model.model.base_model.base_model.base_model.language_model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    if "Qwen" in model.__class__.__name__:
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.text_config.hidden_size), dtype=dtype, device=dev
        )
    else:
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, 'attention_type'):
                self.attention_type = module.attention_type
            else:
                self.attention_type = "causal"
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    model.to(dev)
    for batch in tqdm(dataloader, desc="Processing batches", leave=True):
        try:
            model(batch[0].to(dev, non_blocking=True), use_cache=False)
        except ValueError:
            pass
    model.to('cpu')
    del dataloader
    torch.cuda.empty_cache()
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "Qwen" in args.model:
        model.model.base_model.base_model.base_model.language_model.embed_tokens = model.model.base_model.base_model.base_model.language_model.embed_tokens.cpu()
        model.model.base_model.base_model.base_model.language_model.norm = model.model.base_model.base_model.base_model.language_model.norm.cpu()
    
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    model = model.cpu()
    layers = layers.cpu()
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        if "Qwen" in model.__class__.__name__ and args.mllm_quant:
            assert sens_inputs_text_list is not None and sens_inputs_vl_list is not None, \
            subset = {k: v.cpu() for k, v in subset.items()}
            for name, submodule in subset.items():
                if not isinstance(submodule, nn.Linear):
                    continue
                if (
                    not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
                ) == (not args.invert):
                    continue

                logging.info(f"[MLLM] Layer {i} / {name} - multi-sample Hessian")
                layer = layer.cpu()
                submodule = submodule.to(dev)
                inps = inps.cpu(); outs = outs.cpu()
                torch.cuda.empty_cache()
                model.to(dev)
                if args.batch_norm:
                    S_text = get_layer_hessian_sensitivity_multi_robust(
                    model,
                    sens_inputs_text_list,
                    target_layer=submodule,
                    device=dev,
                    lambda_reg=args.percdamp,
                    max_tokens=1024,
                    )
                    S_vis = get_layer_hessian_sensitivity_multi_robust(
                        model,
                        sens_inputs_vl_list,
                        target_layer=submodule,
                        device=dev,
                        lambda_reg=args.percdamp,
                        max_tokens=1024,
                    )
                else:
                    S_text, cov_trace_text, f_norm_text = get_layer_hessian_sensitivity_multi(
                        model,
                        sens_inputs_text_list,
                        target_layer=submodule,
                        device=dev,
                        lambda_reg=args.percdamp,
                        max_tokens=model.seqlen,
                    )
                    S_text = S_text.cpu()
                    torch.cuda.empty_cache()
                    S_vis, cov_trace_vis, f_norm_vis = get_layer_hessian_sensitivity_multi(
                        model,
                        sens_inputs_vl_list,
                        target_layer=submodule,
                        device=dev,
                        lambda_reg=args.percdamp,
                        max_tokens=model.seqlen,
                    )
                model.to('cpu')
                torch.cuda.empty_cache()
                S_text = S_text.to(dev)
                mask_text, mask_vision, mask_cross, mask_none, stats = modality_structural_partition(
                    S_text,
                    S_vis,
                    salient_top_p=args.salient_top_p,
                    diff_quantile=args.diff_quantile,
                    sparsity_thr=torch.tensor(args.sparsity_thr, device=dev),
                    logger=logger
                )
                if args.adaptive_omega:
                    with torch.no_grad():
                        mask_uni = mask_text | mask_vision
                        uni_sum = (S_text + S_vis)[mask_uni].sum()
                        r_layer = uni_sum / (S_text + S_vis).sum()
                        logger.info(f"r_layer: {r_layer} \n")
                        rho = stats["omega"].to(dev)
                        omega = r_layer * rho + (1 - r_layer) * (1 - rho)
                        stats["omega"] = omega
                submodule.sparsity = stats["omega"]
                submodule.r_layer = r_layer
                if save_folder is not None:
                    safe_layer_name = name.replace(".", "--")
                    cache_save_folder = os.path.join(save_folder, "omega_cache")
                    os.makedirs(cache_save_folder, exist_ok=True)
                    torch.save(stats["omega"].cpu(), os.path.join(cache_save_folder, f"S_hessian_{i}_{safe_layer_name}.pt"))
                del stats["omega"], rho, S_text, S_vis, mask_text, mask_vision, mask_uni, mask_none, mask_cross
                torch.cuda.empty_cache()
        gptq = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            logger.info(f"Method: {args.low_quant_method}")
            subset[name] = subset[name].to(dev)
            if args.low_quant_method == "gptq":
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    3, perchannel=True, sym=False, mse=False, trits=False
                )
            else:
                braq_quantizer = Binarization(
                    subset[name].weight,
                    method=args.low_quant_method,
                    groupsize=groupsize,
                )
                gptq[name] = BRAGPTQ(
                    subset[name],
                    braq_quantizer,
                    salient_metric=args.salient_metric,
                    disable_gptq=args.disable_gptq,
                    method=args.low_quant_method,
                    order2_group=args.order2_group,
                    mllm_quant=args.mllm_quant,
                )
            subset[name].sparsity = None

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp
        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        model = model.cpu()
        layer = layer.to(dev)
        if "Qwen" in model.__class__.__name__:
            rotary_emb = model.model.base_model.base_model.base_model.language_model.rotary_emb
            rotary_emb = rotary_emb.to(dev)
            seq_len = inps[0].size(0) if inps[0].dim() == 2 else inps[0].size(1)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=dev).unsqueeze(0)
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1) if "Qwen2.5" in args.model else position_ids
            for j in range(args.nsamples):
                hidden = inps[j].unsqueeze(0).to(dev)
    
                cos, sin = rotary_emb(hidden, position_ids)
                position_embeddings = (cos, sin)
                outs[j] = layer(  
                    hidden,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    )[0]
                del hidden, cos, sin, position_embeddings
        else:
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gptq:
            logging.info(f'{i} {name}')
            logging.info("Quantizing ...")
            if args.low_quant_method == "gptq":
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=128, actorder=False, static_groups=False
                )
            else:
                info = gptq[name].fasterquant(
                    percdamp=args.percdamp, 
                    blocksize=args.blocksize,
                    num_p=args.num_p,
                )
            gptq[name].free()
        
        
        for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    if "Qwen" in model.__class__.__name__:
        model.config.text_config.use_cache = use_cache
    else:
        model.config.use_cache = use_cache

if __name__ == "__main__":
    import argparse
    from datautils import *

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["arb", 'sab', 'braq', 'xnor','gptq'],
        help="alternating refined binarization method",
    )
    parser.add_argument(
        "--order2_group",
        action='store_true',
        help="division for salient weights",
    )
    parser.set_defaults(order2_group=False)
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--num_p",
        type=int,
        default=1,
        help="Number of division for non-salient weights",
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
    )
    # MLLM
    parser.add_argument(
    "--mllm_quant",
    action="store_true",
    )
    parser.add_argument(
        "--salient_top_p",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--diff_quantile",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--sparsity_thr",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--rc_iter",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--batch_norm",
        action="store_true",
    )
    parser.add_argument(
        "--adaptive_omega",
        action="store_true",
    )

    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()
    groupsize = args.blocksize
    
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    device = args.device
    save_title = f"{args.model.split('/')[-1]}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_nump_{args.num_p}_order2group_{args.order2_group}"
    save_title = f"{args.model.split('/')[-1]}_{args.dataset}_{args.low_quant_method}_{groupsize}_{args.salient_metric}_nump_{args.num_p}_order2group_{args.order2_group}"
    suffix = f"_{timestamp}_adaptive-omega" if args.adaptive_omega else f"_{timestamp}_test"
    save_file = os.path.join("output", save_title.replace("/", "_") + suffix)
    if args.load_quantized:
        model = get_model(save_file)
        model.eval()

    else:
        log_file = "./log/" + save_title.replace("/", "_") + f"_{args.experiment}_{timestamp}" + ".log"
        log_path = os.path.dirname(log_file)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logger = setup_logger(log_file)
        logger.info(f"args: {args}")
        model, processor = get_model(args.model)
        text_hessian_inputs = None
        vision_hessian_inputs = None

        if ("Qwen3-VL" in args.model or "Qwen2.5-VL" in args.model):
            vl_hessian_inputs, text_hessian_inputs, vision_hessian_inputs = build_mixed_inputs(args, processor, device, docvqa_ratio=0.0)    
            mix_loader  = []
            for d in vl_hessian_inputs:
                inps = {k: v for k, v in d.items()}
                tar = d['input_ids'].clone()
                tar[:, :-1] = -100
                mix_loader.append((inps, tar))
        tick = time.time()
        dataloader, textloader = None, None
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        quant_sequential(
            model,
            dataloader,
            device,
            sens_inputs_text_list=text_hessian_inputs,
            sens_inputs_vl_list=vision_hessian_inputs,
            save_folder=save_file,
        )

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)
        try:
            processor.save_pretrained(save_file)
        except:
            pass
        logger.info(f"Model saved to {save_file}")