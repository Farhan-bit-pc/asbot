"""
merge_and_push.py
------------------
ONE-TIME setup step, run locally (not on Netlify) before deploying.

The Netlify Function calls a hosted model over HTTP — it can't load your
local LoRA adapter itself. This script merges the adapter into the base
model's weights and pushes the result to your Hugging Face account, so you
have a model repo you can point HF_API_URL at.

Usage:
    pip install transformers peft torch huggingface_hub
    huggingface-cli login                      # paste a token with write access
    python merge_and_push.py --repo-id your-username/as-chatbot-merged

After this finishes, you have two ways to actually serve it (pick one —
this script doesn't do either, it just gets the weights onto the Hub):

  A) Hugging Face Inference Endpoints (recommended — reliable, works with
     any custom model, autoscales, can scale to zero when idle):
     huggingface.co/your-username/as-chatbot-merged -> "Deploy" -> "Inference Endpoint"
     Then HF_API_URL = the endpoint URL it gives you (looks like
     https://xxxxx.us-east-1.aws.endpoints.huggingface.cloud).

  B) Free Serverless Inference API — HF_API_URL =
     https://api-inference.huggingface.co/models/your-username/as-chatbot-merged
     This is free but not guaranteed to have your specific custom model
     "warm" — small/custom models can 503 with a loading delay on first
     call, and there's no uptime guarantee. Fine for testing, not for
     something you want reliably live.

Either way, set HF_API_TOKEN in Netlify's environment variables to a token
with inference permission.
"""

from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="./as-chatbot-final")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/as-chatbot-merged")
    parser.add_argument("--private", action="store_true", help="push as a private repo (recommended — this model reflects real personal data)")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)

    print(f"Applying LoRA adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging adapter into base weights...")
    merged_model = model.merge_and_unload()

    print(f"Pushing merged model to: {args.repo_id} (private={args.private})")
    merged_model.push_to_hub(args.repo_id, private=args.private)
    tokenizer.push_to_hub(args.repo_id, private=args.private)

    print("Done. Next steps:")
    print(f"  1. Go to https://huggingface.co/{args.repo_id}")
    print("  2. Deploy it (Inference Endpoint recommended — see this file's docstring)")
    print("  3. Set HF_API_URL and HF_API_TOKEN in Netlify's environment variables")


if __name__ == "__main__":
    main()
