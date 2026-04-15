# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Apply Liger kernel optimizations globally before model loading
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2, apply_liger_kernel_to_gemma2
    _liger_registry = {
        "qwen2": apply_liger_kernel_to_qwen2,
        "gemma2": apply_liger_kernel_to_gemma2,
    }
    _liger_applied = False  # applied later once model_id is known
    print("✓ Liger kernels available (will apply after model selection)")
except Exception as e:
    _liger_registry = {}
    print(f"Could not import Liger kernels: {e}")

import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

try:
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
except ImportError:
    Gemma3DecoderLayer = None

from coconut import Coconut
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    get_cot_latent_dataset,
    MyCollator,
)

from tqdm import tqdm
from copy import copy
import itertools
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed
import re
from typing import Optional

# Regex and helpers for extracting boxed answers from model outputs
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)

OPTIONAL_CONFIG_DEFAULTS = {
    "enable_gen_eval": False,
    "eval_do_sample": False,
    "eval_temperature": 0.7,
    "eval_top_p": 0.8,
    "eval_top_k": None,
    "use_chat_template": False,
    "use_boxed_answer": True,
    "answer_prefix": "In summary, ",
    "use_ddp": False,
    "torch_compile": False,
    "save_every": 1,
    "latent_init_token": "<<",
    "smoke_single_gpu": False,
    "train_fraction": None,
    "max_new_tokens": None,
    "max_eval_samples": None,
    "disable_kv_cache": False,
}


def extract_last_boxed(text: str) -> Optional[str]:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return None


def parse_prediction(raw_text: str) -> float:
    """
    Map the model's raw output to a binary prediction.
    Defaults to searching the final boxed answer, then falls back to the raw text.
    Returns 1.0 for affirmative (contains 'yes' and not 'no'), else 0.0.
    """
    candidate = extract_last_boxed(raw_text)
    if not candidate:
        candidate = raw_text or ""
    candidate = candidate.lower()
    return 1.0 if ("yes" in candidate and "no" not in candidate) else 0.0


def safe_int_from_text(text: str) -> Optional[int]:
    """Best-effort to extract an integer from text. Returns None if not found."""
    if text is None:
        return None
    txt = str(text).strip()
    # direct cast
    try:
        return int(txt)
    except Exception:
        pass
    # find last signed integer in the text
    nums = re.findall(r"-?\d+", txt)
    if nums:
        try:
            return int(nums[-1])
        except Exception:
            return None
    return None


def _log_non_default_optional_config(config_dict: dict) -> None:
    changed = {}
    for key, default in OPTIONAL_CONFIG_DEFAULTS.items():
        if key in config_dict and config_dict[key] != default:
            changed[key] = config_dict[key]
    if changed:
        print("Non-default optional config values:")
        for key in sorted(changed.keys()):
            print(f"  {key}: {changed[key]!r} (default: {OPTIONAL_CONFIG_DEFAULTS[key]!r})")


def main():

    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("config_file")
    args = parser.parse_args()

    # init distributed environment
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    if rank == 0:
        print("Config:", config_dict)
        _log_non_default_optional_config(config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)
    if getattr(configs, "smoke_single_gpu", False) and world_size != 1:
        raise ValueError(
            "Smoke config requires a single GPU. Re-run with "
            "`torchrun --nproc_per_node 1` (or use a non-smoke config)."
        )

    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)

    torch.distributed.barrier()
    cur_ckpts = os.listdir(save_dir)

    # check if the job is preempted and resumed.

    if len(cur_ckpts) > 0 and not configs.only_eval:
        # if there are previous checkpoints, and only_eval is False
        # it means the previous run was preempted and the program is restarted.
        # need to find the latest checkpoint and resume from that.

        if rank == 0:
            print(
                f"Warning: found previous run and gonna resume from that. the inputted `resume` argument is ignored!"
            )

        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))

        # Get the last item in the sorted list
        latest_checkpoint = checkpoints[-1] if checkpoints else None
        configs.resume = int(latest_checkpoint.split("_")[1])
        load_dir = os.path.join(configs.save_path, configs.name, latest_checkpoint)

        configs.load_model_path = load_dir
        print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        # by setting `resume`, we can skip a few epoches at the beginning.
        if configs.load_model_path == "None":
            print(
                f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!"
            )
            # not an intended use case at this point
        print(
            f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs"
        )

    # Apply Liger kernel for the right model family before loading
    if _liger_registry:
        model_id_lower = configs.model_id.lower()
        for key, apply_fn in _liger_registry.items():
            if key in model_id_lower:
                apply_fn()
                print(f"✓ Applied Liger kernel optimizations for {key}")
                break

    # Load model with Flash Attention 2 for speed
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if configs.bf16 else torch.float16,
        device_map=None,  # We handle device placement manually
    )
    print("✓ Loaded model with Flash Attention 2")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("✓ Enabled gradient checkpointing")
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    if tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer must define an eos_token for this training pipeline."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side != "right":
        print(
            f"WARNING: tokenizer.padding_side={tokenizer.padding_side!r}; "
            "overriding to 'right' for Coconut collator."
        )
        tokenizer.padding_side = "right"
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    unk_id = tokenizer.unk_token_id
    for tok_name, tok_id in (
        ("<|latent|>", latent_id),
        ("<|start-latent|>", start_id),
        ("<|end-latent|>", end_id),
    ):
        if tok_id is None or tok_id < 0 or (unk_id is not None and tok_id == unk_id):
            raise ValueError(
                f"Special token {tok_name} failed to register correctly in tokenizer."
            )

    loaded = False

    if configs.load_model_path != "None":
        saved_weights = torch.load(
            configs.load_model_path, map_location=torch.device(rank)
        )

        # Normalize potential wrapper prefixes, e.g., DDP adds 'module.'
        if any(k.startswith("module.") for k in saved_weights.keys()):
            saved_weights = {k[len("module."):]: v for k, v in saved_weights.items()}

        # Some compilers/wrappers may add '_orig_mod.'
        if any(k.startswith("_orig_mod.") for k in saved_weights.keys()):
            saved_weights = {
                k[len("_orig_mod."):]: v for k, v in saved_weights.items()
            }

        has_coconut_prefix = any(
            k.startswith("base_causallm") for k in saved_weights.keys()
        )

        if configs.coconut and not has_coconut_prefix:
            # Loading a base LM (e.g., SFT) checkpoint into coconut model.
            # Safe to load into base model before wrapping.
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

        elif not configs.coconut and has_coconut_prefix:
            raise ValueError("Cannot load coconut model weights into a causallm model")

        elif configs.coconut and has_coconut_prefix:
            # Will load into Coconut wrapper after it's constructed below.
            pass

        else:
            # Resume/evaluate base LM checkpoint
            loaded = True
            print(model.load_state_dict(saved_weights, strict=False))

    if not (configs.cot or configs.no_thoughts or configs.no_cot):
        # if we need new tokens, initialize their embeddings and lm heads
        model.resize_token_embeddings(len(tokenizer))
        embeddings = model.get_input_embeddings()
        latent_init_token = getattr(configs, "latent_init_token", "<<")
        target_id = tokenizer.convert_tokens_to_ids(latent_init_token)
        if (
            target_id is None
            or target_id < 0
            or (unk_id is not None and target_id == unk_id)
        ):
            raise ValueError(
                f"Initialization token {latent_init_token!r} is not in the tokenizer vocabulary."
            )
        # initialize the new token embeddings with a known token
        # it helps stablize the training
        for token_id in [latent_id, start_id, end_id]:
            target_embedding = embeddings.weight.data[target_id]
            embeddings.weight.data[token_id] = target_embedding
            # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
            lm_head = model.lm_head
            lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    if configs.no_thoughts:
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id,
                        disable_kv_cache=getattr(configs, "disable_kv_cache", False))

    if configs.load_model_path != "None" and not loaded:
        # At this point, a Coconut-wrapped checkpoint should be loaded into the wrapper
        print(model.load_state_dict(saved_weights, strict=False))

    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)

    _fsdp_layer_cls = {LlamaDecoderLayer}
    if Gemma3DecoderLayer is not None:
        _fsdp_layer_cls.add(Gemma3DecoderLayer)
    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=_fsdp_layer_cls,
    )

    if configs.bf16:
        model.to(torch.bfloat16)

    # Use DDP if specified in config or for eval, otherwise use FSDP
    if configs.only_eval or getattr(configs, 'use_ddp', False):
        parallel_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        print(f"Using DDP on rank {rank}")
    else:
        parallel_model = FSDP(
            model, auto_wrap_policy=llama_auto_wrap_policy, device_id=rank
        )
        print(f"Using FSDP on rank {rank}")

    del model

    # Apply torch.compile if specified
    if getattr(configs, 'torch_compile', False):
        print(f"Applying torch.compile on rank {rank}")
        parallel_model = torch.compile(parallel_model, mode='reduce-overhead')

    if rank == 0:
        print(parallel_model)

    # prepare the ground truth answer and cot for evaluation
    question_val = [d["question"] for d in json.load(open(configs.val_path))]
    answers_val = [
        d["answer"].replace(",", "").strip() for d in json.load(open(configs.val_path))
    ]
    cot_val = ["\n".join(d["steps"]) for d in json.load(open(configs.val_path))]
    if not (len(question_val) == len(answers_val) == len(cot_val)):
        raise ValueError(
            "Validation file length mismatch between questions, answers, and steps. "
            f"Got questions={len(question_val)}, answers={len(answers_val)}, steps={len(cot_val)}."
        )
    if len(question_val) == 0:
        raise ValueError(
            f"Validation file is empty: {configs.val_path}. "
            "Refusing to train without validation data."
        )

    use_chat_template = getattr(configs, "use_chat_template", False)

    max_eval_size = getattr(configs, "max_eval_samples", None)
    if max_eval_size is None:
        max_eval_size = 32 if configs.debug else 100000000
    base_dataset_valid = get_dataset(
        configs.val_path,
        tokenizer,
        max_size=max_eval_size,
        use_chat_template=use_chat_template,
        answer_prefix=getattr(configs, "answer_prefix", "In summary, "),
    )
    if len(base_dataset_valid) == 0:
        raise ValueError(
            f"Tokenized validation dataset is empty: {configs.val_path}. "
            "Refusing to train without validation data."
        )

    if not configs.only_eval:
        train_fraction = getattr(configs, "train_fraction", None)
        if train_fraction is not None:
            if not (0 < train_fraction <= 1.0):
                raise ValueError(
                    f"train_fraction must be in (0, 1], got {train_fraction}."
                )
        max_train_size = 5000 if configs.debug else 100000000
        if train_fraction is not None:
            max_train_size = max(1, int(max_train_size * float(train_fraction)))
        base_dataset_train = get_dataset(
            configs.train_path,
            tokenizer,
            max_size=max_train_size,
            use_chat_template=use_chat_template,
            answer_prefix=getattr(configs, "answer_prefix", "In summary, "),
        )

    if getattr(configs, "max_new_tokens", None) is not None:
        max_new_tokens = configs.max_new_tokens
    elif "gsm" in configs.val_path:
        max_new_tokens = 64
    else:
        max_new_tokens = 2048

    total_train_steps = 0

    if not configs.debug and not configs.only_eval and rank == 0:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    if configs.reset_optimizer:
        optimizer = None

    else:
        optimizer = optim.AdamW(
            parallel_model.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )

    best_acc = 0

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    # When only_eval=True, ensure we run the evaluation exactly once.
    # Using `resume` as the epoch index preserves scheduled_stage behavior.
    if configs.only_eval:
        epoch_iter = [configs.resume]
    else:
        epoch_iter = range(configs.resume, configs.num_epochs)

    for epoch in epoch_iter:

        # For saving evaluation outputs per epoch
        eval_outputs = []

        scheduled_stage = (
            0 if (configs.cot or configs.no_cot) else epoch // configs.epochs_per_stage
        )
        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
        )

        valid_gen_dataloader = torch.utils.data.DataLoader(
            dataset_gen_val,
            num_workers=1,
            pin_memory=True,
            batch_size=1,
            collate_fn=collator,
            sampler=DistributedSampler(dataset_gen_val, shuffle=False),
        )

        if not configs.only_eval:

            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
                shuffle=True,
            )

            train_dataloader = torch.utils.data.DataLoader(
                dataset_train,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_train, shuffle=True),
            )

            # the sampler is deterministic even if shuffle is set to True
            # so we have shuffled the dataset when it's constructed (at every epoch).

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or configs.no_cot or configs.no_thoughts,
            )
            if len(dataset_loss_val) == 0:
                raise ValueError(
                    f"Validation loss dataset is empty at epoch {epoch+1}. "
                    "Check val data and preprocessing."
                )

            valid_loss_dataloader = torch.utils.data.DataLoader(
                dataset_loss_val,
                num_workers=1,
                shuffle=False,
                pin_memory=True,
                batch_size=configs.batch_size_training,
                collate_fn=collator,
                sampler=DistributedSampler(dataset_loss_val, shuffle=False),
            )

            if configs.reset_optimizer:
                del optimizer

                optimizer = optim.AdamW(
                    parallel_model.parameters(),
                    lr=configs.lr,
                    weight_decay=configs.weight_decay,
                )

            parallel_model.module.train()

            total_length = len(train_dataloader) // configs.gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(train_dataloader):

                if step == 0 and wandb_run and rank == 0:
                    print("logging training data")
                    cur_bs = len(batch["input_ids"])
                    text_str = ""
                    for data_idx in range(cur_bs):
                        for token_idx in range(len(batch["input_ids"][data_idx])):
                            text_str += (
                                str(batch["input_ids"][data_idx][token_idx].item())
                                + " "
                                + str(batch["labels"][data_idx][token_idx].item())
                                + " "
                                + tokenizer.decode(
                                    batch["input_ids"][data_idx][token_idx]
                                )
                                + "\n"
                            )
                        text_str += "====" * 10 + "\n"
                    text_table.add_data(total_train_steps, text_str)
                    # copy the table due to a bug in wandb
                    # https://github.com/wandb/wandb/issues/2981

                    wandb_run.log({"data_table": copy(text_table)})

                total_train_steps += 1
                batch = {
                    key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                }

                outputs = parallel_model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and rank == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()
            dist.barrier()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
                save_every = getattr(configs, "save_every", 1)
                if save_every <= 0:
                    raise ValueError(f"save_every must be >= 1, got {save_every}.")
                if (epoch + 1) % save_every == 0:
                    states = parallel_model.state_dict()
                    if rank == 0:
                        torch.save(
                            states, os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                        )
                        print("saving model.")

                    dist.barrier()
                    del states
                    gc.collect()
                    torch.cuda.empty_cache()

            # val loss
            total_loss = 0

            with torch.no_grad():
                parallel_model.module.eval()
                for step, batch in enumerate(valid_loss_dataloader):

                    batch = {
                        key: batch[key].to(rank) for key in batch.keys() if key != "idx"
                    }

                    outputs = parallel_model(**batch)
                    loss = outputs.loss
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    total_loss += loss.item() / world_size

                if wandb_run and rank == 0:

                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

        enable_gen_eval = getattr(configs, "enable_gen_eval", False)
        cor = torch.tensor(0, device=rank)
        total = torch.tensor(0, device=rank)
        gen_token_sum = torch.tensor(0, device=rank, dtype=torch.long)
        accuracy = 0.0

        if enable_gen_eval:
            total_length = len(valid_gen_dataloader)
            pbar = tqdm(
                colour="blue", desc=f"Test Accuracy", total=total_length, dynamic_ncols=True
            )
            with torch.no_grad():
                parallel_model.module.eval()
                for idx, batch in enumerate(valid_gen_dataloader):
                    test_idx = batch["idx"][0]

                    batch = {
                        k: v.to(rank)
                        for k, v in batch.items()
                        if v is not None and k not in ["idx", "position_ids"]
                    }

                    assert len(batch["input_ids"]) == 1
                    answer = answers_val[test_idx.cpu().item()]
                    question = question_val[test_idx.cpu().item()]

                    total += 1

                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "synced_gpus": not configs.only_eval,
                    }
                    if getattr(configs, "eval_do_sample", False):
                        gen_kwargs.update(
                            {
                                "do_sample": True,
                                "temperature": getattr(configs, "eval_temperature", 0.7),
                                "top_p": getattr(configs, "eval_top_p", 0.8),
                                "top_k": getattr(configs, "eval_top_k", None),
                            }
                        )

                    outputs = parallel_model.module.generate(
                        **batch,
                        **gen_kwargs,
                    )

                    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    default_extracted_answer = (
                        text_output.split("#")[-1].replace(",", "").strip()
                    )
                    input_ids = batch["input_ids"][0]
                    num_latent = (input_ids == latent_id).sum().item()
                    gen_len = len(outputs[0]) - len(input_ids)
                    gen_token_sum += gen_len

                    use_boxed = getattr(configs, "use_boxed_answer", True)
                    boxed_extracted = (
                        extract_last_boxed(text_output) if use_boxed else None
                    )
                    answer_output = (
                        boxed_extracted if boxed_extracted else default_extracted_answer
                    )

                    if use_boxed:
                        pred_int = safe_int_from_text(answer_output)
                        if pred_int is None:
                            pred_int = int(parse_prediction(text_output))

                        gt_int = safe_int_from_text(answer)
                        if gt_int is None:
                            gt_int = int(parse_prediction(answer))

                        ans_correct = (
                            pred_int is not None
                            and gt_int is not None
                            and pred_int == gt_int
                        )
                    else:
                        ans_correct = answer_output == answer

                    eval_outputs.append(
                        {
                            "idx": test_idx.cpu().item(),
                            "question": question,
                            "ground_truth_answer": answer,
                            "generated_output": text_output,
                            "extracted_answer": answer_output,
                            "boxed_extracted_answer": boxed_extracted,
                            "answer_correct": ans_correct,
                            "num_latent_tokens": num_latent,
                            "gen_tokens": gen_len,
                        }
                    )

                    if idx < 5 and rank == 0:
                        print(
                            f"Question {test_idx}: Answer = '{answer}'"
                        )
                        print(f"Full output: '{tokenizer.decode(outputs[0])}'")
                        print(f"Extracted Output: '{answer_output}'")

                    cor += 1 if ans_correct else 0

                    pbar.update(1)
                    pbar.set_description(
                        f"Test accuracy: {round(float(cor.detach().float() / total.detach().float()), 2)}"
                    )

                pbar.close()
                print(f"Device {rank}: Cor={cor}, Total={total}")

            dist.all_reduce(cor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            dist.all_reduce(gen_token_sum, op=dist.ReduceOp.SUM)

            cor = cor.item()
            total = total.item()
            gen_sum = gen_token_sum.item()
            if rank == 0:
                accuracy = cor / total if total > 0 else 0
                avg_gen_tokens = (gen_sum / total) if total > 0 else 0
                print(f"Accuracy on validation set: {cor} / {total} = {accuracy}")
                print(f"Eval summary -> accuracy: {accuracy:.4f}, samples: {total}, avg_gen_tokens: {avg_gen_tokens:.2f}")
            sys.stdout.flush()
            if wandb_run:
                wandb_run.log({"eval/acc": accuracy})
        else:
            cor = 0
            total = 0
            if rank == 0:
                print("Skipping generation eval (enable_gen_eval=False).")

        outputs_to_save = None
        if configs.only_eval:
            gathered_eval_outputs = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_eval_outputs, eval_outputs)
            if rank == 0:
                seen_idxs = set()
                deduped = []
                for shard in gathered_eval_outputs:
                    for entry in shard:
                        if entry["idx"] not in seen_idxs:
                            seen_idxs.add(entry["idx"])
                            deduped.append(entry)
                outputs_to_save = deduped

        # Save evaluation outputs to JSON file
        if configs.only_eval and rank == 0:
            if outputs_to_save:
                deduped_total = len(outputs_to_save)
                deduped_cor = sum(1 for e in outputs_to_save if e["answer_correct"])
            else:
                deduped_total = total
                deduped_cor = cor

            output_file = os.path.join(save_dir, "eval_outputs.json")
            with open(output_file, "w") as f:
                json.dump({
                    "config": config_dict,
                    "checkpoint": configs.load_model_path,
                    "accuracy": deduped_cor / deduped_total if deduped_total > 0 else 0,
                    "total_samples": deduped_total,
                    "correct_answers": deduped_cor,
                    "outputs": outputs_to_save if outputs_to_save is not None else eval_outputs
                }, f, indent=2)
            print(f"\n✓ Saved evaluation outputs to: {output_file}")

        if configs.only_eval:
            break

        dist.barrier()
        if (
            total > 0 and
            (cor / total) > best_acc
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            states = parallel_model.state_dict()

            if rank == 0:
                torch.save(states, os.path.join(save_dir, f"checkpoint_{epoch + 1}"))
                print("saving model.")

            best_acc = cor / total if total > 0 else 0

            dist.barrier()
            del states
            gc.collect()
            torch.cuda.empty_cache()


    # Cleanly shut down the process group to avoid warnings on exit
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

if __name__ == "__main__":
    main()
