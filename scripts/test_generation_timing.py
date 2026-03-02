#!/usr/bin/env python
"""
Test script to profile Coconut generation and understand the timing breakdown.

Loads a checkpoint, runs generation on a few samples, and reports:
- Time spent in latent prefill (Coconut.forward)
- Time spent in autoregressive decoding
- Total time per sample
- Number of generated tokens

Usage:
    python scripts/test_generation_timing.py \
        --checkpoint /mnt/disk/coconut/checkpoints/sweep-s3-c1/checkpoint_18 \
        --config args/sweep_s3_c1.yaml \
        --val-path ff_data/val_litereason.json \
        --num-samples 5 \
        --resume 18
"""

import torch
import time
import json
import yaml
import argparse
import sys
import os

sys.path.insert(0, "/mnt/disk/coconut")

from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
from dataset import get_dataset, get_question_latent_dataset, MyCollator
from utils import Config, set_seed


def monkey_patch_generate_with_timing(coconut_model):
    """Wrap Coconut.generate to measure prefill vs decode time."""
    original_forward = coconut_model.forward.__func__
    original_generate = coconut_model.generate.__func__

    def timed_generate(self, input_ids, attention_mask, max_new_tokens=16,
                       output_embedding=False, synced_gpus=False,
                       do_sample=False, temperature=0.7, top_p=0.8, top_k=None,
                       **kwargs):
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1

        tokens = input_ids[0].detach().tolist()
        labels = input_ids.clone()

        # --- Prefill (latent reasoning) ---
        t0 = time.time()
        outputs = original_forward(
            self, input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(0, input_ids.shape[1], dtype=torch.long,
                          device=input_ids.device).reshape(1, -1),
            token_type_ids=torch.zeros_like(input_ids, device=input_ids.device),
        )
        torch.cuda.synchronize()
        prefill_time = time.time() - t0

        inputs_embeds = outputs.inputs_embeds

        # --- Decode ---
        t1 = time.time()
        next_token = self._sample_token(outputs.logits[0, -1], do_sample,
                                         temperature, top_p, top_k)
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        past_key_values = None
        for _ in range(max_new_tokens - 1):
            out = self.base_causallm(
                inputs_embeds=new_inputs_embeds if past_key_values is None else new_token_embed,
                past_key_values=past_key_values,
                use_cache=True,
                token_type_ids=(
                    torch.zeros_like(new_inputs_embeds[..., 0], dtype=torch.long)
                    if past_key_values is None
                    else torch.zeros_like(new_token_embed[..., 0], dtype=torch.long)
                ),
            )
            past_key_values = out.past_key_values
            self.gen_forward_cnt += 1
            next_token = self._sample_token(out.logits[0, -1], do_sample,
                                             temperature, top_p, top_k)
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)

        torch.cuda.synchronize()
        decode_time = time.time() - t1

        num_generated = len(tokens) - input_ids.shape[1]
        self._last_timing = {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_time": prefill_time + decode_time,
            "num_generated_tokens": num_generated,
            "input_tokens": input_ids.shape[1],
        }

        return torch.tensor(tokens).view(1, -1)

    import types
    coconut_model.generate = types.MethodType(timed_generate, coconut_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-path", default="ff_data/val_litereason.json")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--resume", type=int, required=True,
                        help="Epoch number (determines latent stage)")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config_dict["resume"] = args.resume
    config_dict["val_path"] = args.val_path
    config_dict["load_model_path"] = args.checkpoint
    configs = Config(config_dict)
    set_seed(configs.seed)

    device = torch.device("cuda:0")

    # Load model
    print(f"Loading model: {configs.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        configs.model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if configs.bf16 else torch.float16,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    latent_init_token = getattr(configs, "latent_init_token", "<<")
    target_id = tokenizer.convert_tokens_to_ids(latent_init_token)
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    coconut_model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)

    # Load checkpoint weights
    print(f"Loading checkpoint: {args.checkpoint}")
    saved_weights = torch.load(args.checkpoint, map_location="cpu")
    if any(k.startswith("module.") for k in saved_weights):
        saved_weights = {k[len("module."):]: v for k, v in saved_weights.items()}
    if any(k.startswith("_orig_mod.") for k in saved_weights):
        saved_weights = {k[len("_orig_mod."):]: v for k, v in saved_weights.items()}
    print(coconut_model.load_state_dict(saved_weights, strict=False))

    coconut_model = coconut_model.to(device)
    coconut_model.eval()
    if configs.bf16:
        coconut_model.to(torch.bfloat16)

    monkey_patch_generate_with_timing(coconut_model)

    # Prepare dataset
    use_chat_template = getattr(configs, "use_chat_template", False)
    base_dataset = get_dataset(args.val_path, tokenizer, max_size=args.num_samples,
                               use_chat_template=use_chat_template)

    scheduled_stage = args.resume // configs.epochs_per_stage
    num_latent_tokens = min(configs.max_latent_stage, scheduled_stage) * configs.c_thought
    print(f"\nScheduled stage: {scheduled_stage}, latent tokens: {num_latent_tokens}")
    print(f"c_thought={configs.c_thought}, max_latent_stage={configs.max_latent_stage}")
    print(f"max_new_tokens={getattr(configs, 'max_new_tokens', 1024)}")

    dataset_gen = get_question_latent_dataset(
        scheduled_stage, base_dataset, configs, start_id, latent_id, end_id,
        no_special_marker=False,
    )

    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    dataloader = torch.utils.data.DataLoader(
        dataset_gen, batch_size=1, collate_fn=collator, shuffle=False,
    )

    val_data = json.load(open(args.val_path))
    max_new_tokens = getattr(configs, "max_new_tokens", 1024)

    # Run generation with timing
    print(f"\n{'='*70}")
    print(f"Running generation on {min(args.num_samples, len(dataset_gen))} samples")
    print(f"{'='*70}\n")

    timings = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= args.num_samples:
                break

            test_idx = batch["idx"][0].item()
            batch_gpu = {
                k: v.to(device) for k, v in batch.items()
                if v is not None and k not in ["idx", "position_ids"]
            }

            outputs = coconut_model.generate(
                **batch_gpu, max_new_tokens=max_new_tokens, synced_gpus=False,
            )

            timing = coconut_model._last_timing
            timings.append(timing)

            text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = tokenizer.decode(
                outputs[0][batch_gpu["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            gt_answer = val_data[test_idx]["answer"] if test_idx < len(val_data) else "?"

            print(f"--- Sample {idx} (data idx={test_idx}) ---")
            print(f"  Input tokens:     {timing['input_tokens']}")
            print(f"  Generated tokens: {timing['num_generated_tokens']}")
            print(f"  Prefill time:     {timing['prefill_time']:.3f}s")
            print(f"  Decode time:      {timing['decode_time']:.3f}s")
            print(f"  Total time:       {timing['total_time']:.3f}s")
            if timing['num_generated_tokens'] > 0:
                print(f"  Decode tok/s:     {timing['num_generated_tokens'] / timing['decode_time']:.1f}")
            print(f"  Ground truth:     {gt_answer}")
            print(f"  Generated:        {generated_text[:200]}")
            print()

    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    avg_prefill = sum(t["prefill_time"] for t in timings) / len(timings)
    avg_decode = sum(t["decode_time"] for t in timings) / len(timings)
    avg_total = sum(t["total_time"] for t in timings) / len(timings)
    avg_gen_tokens = sum(t["num_generated_tokens"] for t in timings) / len(timings)
    print(f"  Avg prefill time:     {avg_prefill:.3f}s ({avg_prefill/avg_total*100:.1f}%)")
    print(f"  Avg decode time:      {avg_decode:.3f}s ({avg_decode/avg_total*100:.1f}%)")
    print(f"  Avg total time:       {avg_total:.3f}s")
    print(f"  Avg generated tokens: {avg_gen_tokens:.1f}")
    print(f"  Num latent tokens:    {num_latent_tokens}")
    print()
    print("Implications:")
    if avg_prefill > avg_decode:
        print("  -> Prefill-dominated: latent processing is the bottleneck.")
        print("     Batching decode won't help much. Need to optimize latent forward pass.")
    else:
        print("  -> Decode-dominated: autoregressive decoding is the bottleneck.")
        print("     Batching could help (but requires rewriting Coconut.generate).")
        print("     Lowering max_new_tokens would help if many tokens are wasted.")


if __name__ == "__main__":
    main()
