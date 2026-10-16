# PAO Memory System

A single-file **Person–Action–Object (PAO)** memory trainer for the numbers **00–99**, with an image gallery and local Stable Diffusion image tooling.

## Live site (GitHub Pages)

- **Landing page:** https://jackeown.github.io/pao-memory-system/
- **Memory helper:** https://jackeown.github.io/pao-memory-system/pao_memory_system_helper.html
- **Image gallery:** https://jackeown.github.io/pao-memory-system/pao_gallery.html

The whole app is static — no build step, no backend.

## What's inside

| File | Description |
| --- | --- |
| `index.html` | Landing page linking to the helper and gallery. |
| `pao_memory_system_helper.html` | The trainer: editable 00–99 sets, duplicate detection, file-drift indicator, digit→picture converter, and spaced-repetition practice. |
| `pao_gallery.html` | All 100 images in one big scroll, grouped into collapsible sections. |
| `generate_images.py` | Generates the 100 mnemonic images with **local** Stable Diffusion (Automatic1111 / Forge / SD.Next, or `diffusers`). |
| `image_prompts.json` / `.txt` | The 100 image prompts derived from the current data. |
| `images/` | Where generated images live, named `00.png` … `99.png` (`.jpg`/`.webp` also work). |

## Memory helper features

- **10 themed sets** of 10 numbers each (00–99), each with a person, an action, and an object.
- **Editable** — every field is a text input; changes save to `localStorage` automatically.
- **Duplicate detection** — flags repeated people/actions/objects (case-insensitive) with a summary banner and clickable number chips that jump to the row.
- **File-drift indicator** — shows how far your saved data has diverged from the in-file defaults, with a detail modal.
- **Digit→picture converter** — type any string of digits; every 6 become one image (2 person / 2 action / 2 object), colour-coded.
- **Practice mode** — spaced-repetition (SM-2 style) with two independent choices:
  - **Question:** *Parts* (person/action/object individually, adaptive multiple-choice or typed) or *Whole image* (pick the correct full triple).
  - **Numbers:** *All numbers* (uses the due/new schedule) or *Selected few* (drill an explicit list, e.g. `5, 27, 84`).
  - Progress persists in `localStorage`; includes due/new/learning/mature stats and overall accuracy.

## Gallery

`pao_gallery.html` shows all 100 images in a single scrolling page, split into 10 collapsible sections (with Expand/Collapse all). Each card overlays the number badge and shows the person/action/object. Missing images render a placeholder, and the header shows how many images were found. It reads your live `pao_memory_data` from `localStorage`, falling back to the embedded snapshot.

## Generating the images

Images are produced locally; nothing is uploaded. See the docstring in `generate_images.py` for details.

### Automatic1111 / Forge / SD.Next (recommended)

1. Install [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) (one-click script) or Automatic1111.
2. Put a cartoon checkpoint in `models/Stable-diffusion/` (e.g. DreamShaper, Anything-v5, toonyou).
3. Launch with the API enabled: `./webui.sh --api`
4. From this repo:

   ```bash
   python3 generate_images.py --check          # confirm the server is reachable
   python3 generate_images.py                  # generate all 100 (skips existing)
   python3 generate_images.py --only 20,21,25 --force   # redo specific numbers
   ```

### diffusers (no server)

```bash
pip install torch diffusers accelerate transformers
python3 generate_images.py --provider diffusers --model stabilityai/sdxl-turbo
```

Useful flags: `--generic` (strip real names when a checkpoint refuses them), `--dry-run`, `--limit`, `--force`, `--with-number`.

> Note: text and numbers render poorly in diffusion models, so the prompts ask for **no text** and the gallery draws the number as an overlay instead.

## Local use

Open `index.html` (or either page) directly in a browser — they work from `file://` too. Image paths in the gallery are relative, so keep `images/` next to the HTML.

## Privacy note

The 00–09 set is personal (family members) and 10–19 contains friends'/colleagues' names. This repository is public, so that data is visible in `pao_memory_system_helper.html`.
