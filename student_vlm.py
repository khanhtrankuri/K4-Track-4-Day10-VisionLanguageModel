"""Student implementation for the Day 10 Qwen3-VL road-damage lab.

The lightweight helpers in this module can be imported without torch. Heavy ML
dependencies are imported lazily so label conversion tests also run on CPU.
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from pathlib import Path

CLASSES = ("pothole", "crack", "manhole")
PROMPT_NORM1000 = (
    "Detect every road-surface pothole, crack, and manhole in this image. "
    "Return only a JSON list. Each item must have exactly the keys label and "
    "bbox_2d, where label is one of pothole, crack, manhole and bbox_2d is "
    "[x1,y1,x2,y2] using integer coordinates on a 0-1000 scale. "
    "Return [] when no object is present. Do not add Markdown or explanation."
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _round_half_up(value):
    """Round non-negative coordinates deterministically (unlike banker's round)."""
    return int(math.floor(value + 0.5))


def yolo_to_box(label, yolo, coord="norm1000", width=None, height=None):
    """Convert normalized YOLO ``xc,yc,w,h`` to the requested xyxy JSON box."""
    if label not in CLASSES:
        raise ValueError(f"unknown label: {label}")
    if len(yolo) != 4 or not all(math.isfinite(float(v)) for v in yolo):
        raise ValueError(f"invalid YOLO box: {yolo!r}")
    xc, yc, w, h = (float(v) for v in yolo)
    if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)):
        raise ValueError(f"YOLO values must be in [0,1]: {yolo!r}")
    if coord == "norm1000":
        sx = sy = 1000.0
    elif coord == "pixel":
        if not width or not height:
            raise ValueError("width and height are required for pixel coordinates")
        sx, sy = float(width), float(height)
    else:
        raise ValueError("coord must be 'norm1000' or 'pixel'")
    x1 = _round_half_up(max(0.0, min(sx, (xc - w / 2) * sx)))
    y1 = _round_half_up(max(0.0, min(sy, (yc - h / 2) * sy)))
    x2 = _round_half_up(max(0.0, min(sx, (xc + w / 2) * sx)))
    y2 = _round_half_up(max(0.0, min(sy, (yc + h / 2) * sy)))
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, coord="norm1000", width=None, height=None):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda o: (o["bbox_2d"][1], o["bbox_2d"][0]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def prompt_for(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        return PROMPT_NORM1000
    if coord != "pixel":
        raise ValueError("coord must be 'norm1000' or 'pixel'")
    return (
        "Detect every road-surface pothole, crack, and manhole in this image. "
        "Return only a JSON list with label and bbox_2d=[x1,y1,x2,y2]. Labels "
        f"must be pothole, crack, or manhole. Use integer pixel coordinates for "
        f"this {width}x{height} image. Return [] when empty; no other text."
    )


def image_path(record, data_dir="data"):
    path = Path(record["image_path"])
    return path if path.is_absolute() else Path(data_dir) / path


def split_summary(splits_dir="data/splits"):
    result = {}
    sessions = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(Path(splits_dir) / f"{split}.jsonl")
        counts = {c: 0 for c in CLASSES}
        for row in rows:
            sessions.setdefault(row["session"], []).append(split)
            for obj in row["objects"]:
                counts[obj["label"]] += 1
        result[split] = {"images": len(rows), "boxes": counts}
    leaked = {s: sorted(set(v)) for s, v in sessions.items() if len(set(v)) > 1}
    if leaked:
        raise AssertionError(f"sessions occur in multiple splits: {leaked}")
    return result


def show_examples(records, data_dir="data", n=6, seed=42):
    """Draw ground-truth boxes and return a matplotlib figure."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    chosen = random.Random(seed).sample(records, min(n, len(records)))
    cols = 3
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    for ax, rec in zip(axes.flat, chosen):
        im = Image.open(image_path(rec, data_dir)).convert("RGB")
        ax.imshow(im)
        for obj in rec["objects"]:
            xc, yc, w, h = obj["yolo"]
            x1, y1 = (xc - w / 2) * rec["width"], (yc - h / 2) * rec["height"]
            rect = patches.Rectangle((x1, y1), w * rec["width"], h * rec["height"],
                                     fill=False, edgecolor="lime", linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 2), obj["label"], color="black",
                    bbox={"facecolor": "lime", "alpha": 0.8, "pad": 1})
        ax.set_title(rec["image_id"])
        ax.axis("off")
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_predictions(records, pred_path, data_dir="data", n=6, coord="norm1000"):
    """Draw a balanced set of successful/error predictions for report evidence."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output

    raw_by_id = {row["image_id"]: row["raw"] for row in read_jsonl(pred_path)}
    ranked = []
    for rec in records:
        parsed = parse_output(raw_by_id.get(rec["image_id"]))
        pred = _to_norm1000(parsed or [], coord, rec["width"], rec["height"])
        per, _ = _match(_gt_boxes(rec), pred)
        tp = sum(v[0] for v in per.values())
        fp = sum(v[1] for v in per.values())
        fn = sum(v[2] for v in per.values())
        ranked.append((rec, pred, tp, fp, fn, parsed is not None))
    good = [x for x in ranked if x[2] > 0 and x[3] == 0 and x[4] == 0]
    errors = [x for x in ranked if x[3] + x[4] > 0 or not x[5]]
    chosen = good[:n // 2] + errors[:n - min(n // 2, len(good))]
    if len(chosen) < n:
        used = {x[0]["image_id"] for x in chosen}
        chosen += [x for x in ranked if x[0]["image_id"] not in used][:n - len(chosen)]
    cols, rows = 3, math.ceil(len(chosen) / 3)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), squeeze=False)
    for ax, (rec, pred, tp, fp, fn, valid) in zip(axes.flat, chosen):
        ax.imshow(Image.open(image_path(rec, data_dir)).convert("RGB"))
        for obj in rec["objects"]:
            xc, yc, w, h = obj["yolo"]
            x1, y1 = (xc - w / 2) * rec["width"], (yc - h / 2) * rec["height"]
            ax.add_patch(patches.Rectangle((x1, y1), w * rec["width"], h * rec["height"],
                                           fill=False, edgecolor="lime", linewidth=2,
                                           linestyle="--"))
        for obj in pred:
            x1, y1, x2, y2 = obj["bbox"]
            x1, x2 = x1 / 1000 * rec["width"], x2 / 1000 * rec["width"]
            y1, y2 = y1 / 1000 * rec["height"], y2 / 1000 * rec["height"]
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor="red", linewidth=2))
            ax.text(x1, max(0, y1 - 2), obj["label"], color="white",
                    bbox={"facecolor": "red", "alpha": 0.7, "pad": 1})
        ax.set_title(f"{rec['image_id']}  TP={tp} FP={fp} FN={fn} JSON={valid}")
        ax.axis("off")
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Ground truth: green dashed | prediction: red", fontsize=13)
    fig.tight_layout()
    return fig


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )


def load_base_model(model_id="Qwen/Qwen3-VL-4B-Instruct", training=False):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the 4-bit Qwen3-VL model")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config(),
        dtype=torch.float16,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left" if not training else "right"
    if training:
        model.config.use_cache = False
    return model, processor


def _user_message(image, prompt):
    return {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}


def encode_example(processor, record, data_dir="data", coord="norm1000"):
    """Encode one SFT sample and mask every token except the assistant answer."""
    import torch
    from PIL import Image

    image = Image.open(image_path(record, data_dir)).convert("RGB")
    prompt = prompt_for(coord, record["width"], record["height"])
    answer = objects_to_answer(record["objects"], coord, record["width"], record["height"])
    user = _user_message(image, prompt)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kwargs)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kwargs)
    n_prompt = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][:, :n_prompt], prompt_enc["input_ids"]):
        raise AssertionError("prompt tokens are not a prefix of the full conversation")
    labels = full_enc["input_ids"].clone()
    labels[:, :n_prompt] = -100
    full_enc["labels"] = labels
    return {key: value.squeeze(0) if key in {"input_ids", "attention_mask", "mm_token_type_ids", "labels"}
            else value for key, value in full_enc.items()}


class RoadDamageDataset:
    def __init__(self, records, processor, data_dir="data", coord="norm1000"):
        self.records = records
        self.processor = processor
        self.data_dir = data_dir
        self.coord = coord

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return encode_example(self.processor, self.records[index], self.data_dir, self.coord)


class VlmCollator:
    """Right-pad text fields; concatenate Qwen's variable-length vision fields."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        batch = {
            "input_ids": pad_sequence([x["input_ids"] for x in examples], True,
                                      padding_value=self.pad_token_id),
            "attention_mask": pad_sequence([x["attention_mask"] for x in examples], True,
                                            padding_value=0),
            "labels": pad_sequence([x["labels"] for x in examples], True, padding_value=-100),
        }
        if "mm_token_type_ids" in examples[0]:
            batch["mm_token_type_ids"] = pad_sequence(
                [x["mm_token_type_ids"] for x in examples], True, padding_value=0)
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [x[key] for x in examples if key in x]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch


LLM_MODULE = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
VIT_MODULE = re.compile(
    r"(?:^|\.)visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))$"
)


def find_lora_targets(model, include_vit=False):
    matched = []
    for name, module in model.named_modules():
        if LLM_MODULE.search(name) or (include_vit and VIT_MODULE.search(name)):
            if module.__class__.__name__ not in {"Linear", "Linear4bit"}:
                raise TypeError(f"LoRA target is not linear: {name} ({type(module).__name__})")
            matched.append(name)
    expected = 348 if include_vit else 252
    if len(matched) != expected:
        raise AssertionError(f"matched {len(matched)} LoRA modules, expected {expected}")
    return matched


def _is_projector(name):
    return "visual.merger" in name or "visual.deepstack_merger_list" in name


def apply_setup(model, setup, rank=16, alpha=None, dropout=0.05):
    """Freeze the base, attach exact LoRA targets, and optionally open projector."""
    from peft import LoraConfig, get_peft_model

    setup = setup.upper()
    if setup not in {"E1", "E2", "E3"}:
        raise ValueError("setup must be E1, E2, or E3")
    # Do this explicitly instead of prepare_model_for_kbit_training(): that helper
    # casts all non-quantized fp16 weights to fp32, while this lab must keep the
    # full ViT in fp16 and cast only the trainable projector to fp32.
    for param in model.parameters():
        param.requires_grad_(False)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    targets = find_lora_targets(model, include_vit=setup == "E3")
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha or 2 * rank, lora_dropout=dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    ))
    if setup in {"E2", "E3"}:
        for name, param in model.named_parameters():
            if _is_projector(name) and "lora_" not in name:
                param.data = param.data.float()
                param.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projector = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and _is_projector(n) and "lora_" not in n)
    return model, {"matched_modules": len(targets), "trainable_params": trainable,
                   "projector_params": projector}


def save_checkpoint(model, out_dir, run_info=None):
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter", safe_serialization=True)
    extra = {name: p.detach().cpu() for name, p in model.named_parameters()
             if _is_projector(name) and "lora_" not in name and p.requires_grad}
    if extra:
        torch.save(extra, out / "extra_trainable.pt")
    if run_info is not None:
        (out / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")


def load_checkpoint(model, checkpoint_dir):
    import torch
    from peft import PeftModel

    checkpoint = Path(checkpoint_dir)
    model = PeftModel.from_pretrained(model, checkpoint / "adapter")
    extra_path = checkpoint / "extra_trainable.pt"
    if extra_path.exists():
        state = torch.load(extra_path, map_location="cpu", weights_only=True)
        current = dict(model.named_parameters())
        missing = [name for name in state if name not in current]
        if missing:
            raise KeyError(f"projector keys do not match model: {missing[:3]}")
        with torch.no_grad():
            for name, value in state.items():
                current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
    return model


def train(model, processor, records, out_dir, *, data_dir="data", coord="norm1000",
          lr=2e-4, batch_size=2, grad_accum=4, max_steps=300, seed=42,
          save_every=100, setup_info=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    model.config.use_cache = False
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dataset = RoadDamageDataset(records, processor, data_dir, coord)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                        collate_fn=VlmCollator(processor.tokenizer.pad_token_id), num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, round(max_steps * 0.05)),
        num_training_steps=max_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses, start, step, iterator = [], time.perf_counter(), 0, iter(loader)
    while step < max_steps:
        running = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimizer step {step}: {loss.item()}")
            running += loss.detach().item()
            scaler.scale(loss / grad_accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        losses.append(running / grad_accum)
        if step == 1 or step % 10 == 0:
            print(f"step {step:03d}/{max_steps} loss={losses[-1]:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        if step % save_every == 0 or step == max_steps:
            info = dict(setup_info or {})
            info.update({"steps": step, "last_loss": losses[-1]})
            save_checkpoint(model, Path(out_dir) / f"step-{step:03d}", info)
    minutes = (time.perf_counter() - start) / 60
    info = dict(setup_info or {})
    info.update({
        "trainable_params": sum(p.numel() for p in params),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "train_minutes": minutes,
        "steps": max_steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    })
    save_checkpoint(model, out_dir, info)
    return info


def predict(model, processor, records, out_path, *, data_dir="data", coord="norm1000",
            batch_size=4, max_new_tokens=256):
    import torch
    from PIL import Image

    processor.tokenizer.padding_side = "left"
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        conversations = []
        for rec in chunk:
            image = Image.open(image_path(rec, data_dir)).convert("RGB")
            conversations.append([_user_message(
                image, prompt_for(coord, rec["width"], rec["height"]))])
        inputs = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        prompt_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts = processor.batch_decode(generated[:, prompt_width:], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
        rows.extend({"image_id": rec["image_id"], "raw": text.strip()}
                    for rec, text in zip(chunk, texts))
        write_jsonl(out_path, rows)  # resumable artifact after every batch
        print(f"predicted {len(rows)}/{len(records)}")
    return rows


def train_fraction(records, fraction=1.0, seed=42):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    if fraction == 1:
        return list(records)
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    keep = set(indices[:max(1, round(len(records) * fraction))])
    return [row for i, row in enumerate(records) if i in keep]


def run_sft_experiment(*, name, setup, rank=16, train_records=None, test_records=None,
                       data_dir="data", work_dir="work", evidence_dir="report/evidence",
                       train_fraction_value=1.0, coord="norm1000", batch_size=2,
                       grad_accum=4, max_steps=300, seed=42):
    """Run one isolated train→predict→score job and return its evidence dict."""
    import gc
    import subprocess
    import sys
    import torch

    train_records = train_records or read_jsonl(Path(data_dir) / "splits/train.jsonl")
    test_records = test_records or read_jsonl(Path(data_dir) / "splits/test.jsonl")
    selected = train_fraction(train_records, train_fraction_value, seed)
    slug = name.lower()
    checkpoint = Path(work_dir) / slug
    pred_path = Path(work_dir) / f"{slug}.jsonl"
    evidence_path = Path(evidence_dir) / f"{slug}.json"
    model, processor = load_base_model(training=True)
    model, setup_info = apply_setup(model, setup, rank=rank)
    run_info = train(
        model, processor, selected, checkpoint, data_dir=data_dir, coord=coord,
        batch_size=batch_size, grad_accum=grad_accum, max_steps=max_steps,
        seed=seed, setup_info=setup_info,
    )
    processor.tokenizer.padding_side = "left"
    predict(model, processor, test_records, pred_path, data_dir=data_dir, coord=coord)
    command = [
        sys.executable, "scripts/score.py", "--pred", str(pred_path),
        "--name", name, "--out", str(evidence_path), "--coord", coord,
    ]
    for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
        command += ["--meta", f"{key}={run_info[key]}"]
    subprocess.run(command, check=True)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return evidence

