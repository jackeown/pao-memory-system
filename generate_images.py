#!/usr/bin/env python3
"""
Generate PAO mnemonic images from image_prompts.json using LOCAL Stable Diffusion.

Providers
  a1111     Automatic1111 / Forge / SD.Next web API at --url  (default, recommended)
  diffusers Run a model directly in Python (no server; needs torch + diffusers)

Outputs images/00.png ... images/99.png next to this script.
The number is NOT drawn inside the image; the gallery overlays it as a badge.

Quick start (A1111/Forge, one-time):
  1. Install Forge:  https://github.com/lllyasviel/stable-diffusion-webui-forge  (one-click script)
  2. Launch with the API:  ./webui.sh --api            (Windows: webui-user.bat with --api)
  3. Put a cartoon model in models/Stable-diffusion/ (e.g. DreamShaper, Anything-v5, toonyou)
  4. Run:  python3 generate_images.py --check          # confirm the server is reachable
           python3 generate_images.py                  # generate everything missing

Examples
  python3 generate_images.py --limit 5 --dry-run
  python3 generate_images.py --only 20,21,25          # regenerate specific numbers
  python3 generate_images.py --generic                # strip real names (if a checkpoint can't handle them)
  python3 generate_images.py --with-number            # also ask the model to draw the number (SD does this poorly)

Diffusers (no server):
  pip install torch diffusers accelerate transformers
  python3 generate_images.py --provider diffusers --model stabilityai/sdxl-turbo
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "image_prompts.json")
OUT_DIR = os.path.join(HERE, "images")

DEFAULT_NEGATIVE = (
    "text, letters, numbers, watermark, signature, logo, caption, "
    "blurry, low quality, deformed, mutated, extra limbs, extra fingers, "
    "multiple subjects, cluttered background, photorealistic, 3d render"
)


def parse_numbers(value):
    if not value:
        return None
    nums = set()
    for tok in value.replace(",", " ").split():
        tok = tok.strip().lstrip("0") or "0"
        if tok.isdigit():
            nums.add(f"{int(tok):02d}")
    return nums


def build_prompt(entry, generic, with_number):
    if generic:
        scene = f"a person {entry['action'].lower()} a {entry['object'].lower()}"
    else:
        scene = f"{entry['person']} {entry['action'].lower()} {entry['object'].lower()}"
    p = (
        f"{scene}. A vivid, memorable picture for a memory-training (PAO) flashcard. "
        f"Style: flat 2D cartoon illustration, bold clean outlines, bright saturated colors, "
        f"simple minimal background, centered single subject, high contrast, mnemonic flashcard style."
    )
    if with_number:
        p += f' The number "{entry["num"]}" appears clearly in a corner as a small clean label.'
    else:
        p += " No text, letters, numbers or watermark anywhere in the image."
    return p


# --- A1111 / Forge / SD.Next ------------------------------------------------
def a1111_check(args):
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/sdapi/v1/options", timeout=10) as r:
            opts = json.loads(r.read())
        print("Reachable:", args.url)
        print("Loaded model:", opts.get("sd_model_checkpoint", "?"))
        return True
    except Exception as exc:  # noqa: BLE001
        print("Not reachable at", args.url, "->", exc, file=sys.stderr)
        return False


def a1111_set_model(args):
    if not args.model:
        return
    payload = json.dumps({"sd_model_checkpoint": args.model}).encode()
    req = urllib.request.Request(
        args.url.rstrip("/") + "/sdapi/v1/options",
        data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    print("Switched checkpoint to:", args.model)


def make_a1111(args):
    base = args.url.rstrip("/")

    def generate(prompt, path):
        payload = {
            "prompt": prompt,
            "negative_prompt": args.negative,
            "steps": args.steps,
            "cfg_scale": args.cfg,
            "width": args.width,
            "height": args.height,
            "sampler_name": args.sampler,
            "seed": args.seed,
            "batch_size": 1,
            "n_iter": 1,
        }
        req = urllib.request.Request(
            base + "/sdapi/v1/txt2img",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            data = json.loads(r.read())
        raw = base64.b64decode(data["images"][0].split(",", 1)[-1])
        with open(path, "wb") as f:
            f.write(raw)

    return generate


# --- diffusers (no server) --------------------------------------------------
def make_diffusers(args):
    import torch
    from diffusers import AutoPipelineForText2Image

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model,
        torch_dtype=dtype,
        variant="fp16" if use_cuda else None,
    )
    pipe = pipe.to("cuda" if use_cuda else "cpu")
    gen_kwargs = {"num_inference_steps": args.steps, "guidance_scale": args.cfg}
    if args.seed is not None and args.seed >= 0:
        gen_kwargs["generator"] = torch.Generator(device=pipe.device).manual_seed(args.seed)
    print(f"Loaded {args.model} on {'cuda' if use_cuda else 'cpu'} "
          f"(steps={args.steps}, cfg={args.cfg})")

    def generate(prompt, path):
        img = pipe(
            prompt=prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            **gen_kwargs,
        ).images[0]
        img.save(path)

    return generate


def main():
    ap = argparse.ArgumentParser(description="Generate PAO images with local Stable Diffusion.")
    ap.add_argument("--provider", choices=["a1111", "diffusers"], default="a1111")
    ap.add_argument("--url", default="http://127.0.0.1:7860", help="A1111/Forge base url")
    ap.add_argument("--model", default=None,
                    help="A1111: checkpoint name; diffusers: HF repo id (default stabilityai/sdxl-turbo)")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--sampler", default="DPM++ 2M Karras")
    ap.add_argument("--seed", type=int, default=-1, help="-1 = random")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only", help="comma-separated numbers, e.g. 5,20,99")
    ap.add_argument("--limit", type=int, default=0, help="max images this run (0 = all)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ap.add_argument("--generic", action="store_true", help="strip real names from prompts")
    ap.add_argument("--with-number", action="store_true", help="ask the model to draw the number")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="test the A1111 server and exit")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    args = ap.parse_args()

    # provider-aware defaults
    turbo = args.provider == "diffusers" and args.model in (None, "", "stabilityai/sdxl-turbo")
    if args.provider == "diffusers" and not args.model:
        args.model = "stabilityai/sdxl-turbo"
    if args.steps is None:
        args.steps = 4 if turbo else 30
    if args.cfg is None:
        args.cfg = 0.0 if turbo else 7.0
    if args.width is None:
        args.width = 768
    if args.height is None:
        args.height = 768

    if args.check:
        sys.exit(0 if a1111_check(args) else 1)

    if not os.path.exists(PROMPTS):
        sys.exit(f"Missing {PROMPTS}")
    with open(PROMPTS, encoding="utf-8") as f:
        entries = json.load(f)

    only = parse_numbers(args.only)
    os.makedirs(OUT_DIR, exist_ok=True)

    todo = []
    for e in entries:
        if only and e["num"] not in only:
            continue
        path = os.path.join(OUT_DIR, e["num"] + ".png")
        if os.path.exists(path) and not args.force:
            continue
        todo.append(e)
    if args.limit:
        todo = todo[: args.limit]

    if args.dry_run:
        for i, e in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {e['num']}: {build_prompt(e, args.generic, args.with_number)}")
        print(f"\n{len(todo)} image(s) would be generated via {args.provider}.")
        return

    if not todo:
        print("Nothing to do (all images present). Use --force to regenerate.")
        return

    if args.provider == "a1111":
        if not a1111_check(args):
            sys.exit("Start your SD web UI with --api (e.g. ./webui.sh --api) and retry.")
        a1111_set_model(args)
        generate = make_a1111(args)
    else:
        generate = make_diffusers(args)

    print(f"Generating {len(todo)} image(s) via {args.provider}...")
    ok = 0
    for i, e in enumerate(todo, 1):
        path = os.path.join(OUT_DIR, e["num"] + ".png")
        prompt = build_prompt(e, args.generic, args.with_number)
        try:
            generate(prompt, path)
            ok += 1
            print(f"[{i}/{len(todo)}] saved {os.path.relpath(path, HERE)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] FAILED {e['num']}: {exc}", file=sys.stderr)
        if args.delay:
            time.sleep(args.delay)

    print(f"Done. {ok}/{len(todo)} saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
