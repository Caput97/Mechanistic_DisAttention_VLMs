# Mechanistic_DisAttention_VLMs


## ▶️ Running the experiments

The main entry point is `run.py`.

### Modes
- `prob`: track layer-wise sentence probabilities under attention knockout.
- `attention_weights`: extract and analyze attention distributions under attention knockout.

### Video modes (`--video_mode`)
- `video`: uses the original per-item videos from the dataset.
- `black_video`: uses a fixed black video (`black_video.mp4`) for all items (control condition).

### Masked regions (`--mask_regions`)
- `all_regions`: evaluates the full default set of masking conditions (including `none` baseline).
- `none`: baseline only (no masking).
- Or a comma-separated list among: `none,user_role,user_text,video,A_content,B_content,video1half,video2half`.

If `--mask_regions` is omitted, it behaves like `--mask_regions all_regions`.

---

## ✅ Probability mode (`prob`) — assistant content

### 1) All regions + per-item videos

```bash
python run.py prob \
  --query_scope assistant_content \
  --video_mode video \
  --mask_regions all_regions
```

### 2) no masking condition, video region and A_content masked + only black videos

```bash
python run.py prob \
  --query_scope assistant_content \
  --video_mode black_video \
  --mask_regions none,video,A_content
```

## ✅ Attention scores retrieval (`attention_weights`) — assistant content

```bash
python run.py attention_weights \
  --query_scope assistant_content \
  --video_mode video \
  --mask_regions all_regions
```

```bash
python run.py attention_weights \
  --query_scope assistant_content \
  --video_mode video \
  --mask_regions video,user_text
```
