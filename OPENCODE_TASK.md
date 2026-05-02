# Task: Fix modal_train.py Stage 2 Training Failures

## Context
We're fine-tuning Kokoro TTS on Modal.com (A100-80GB). Stage 1 completed successfully. Stage 2 keeps crashing with different errors. The root cause is fragile patching logic in `scripts/modal_train.py` for `models.py` on the Modal volume.

## Problems to Fix

### Problem 1: Import verification fails in `setup_training()` (line 660-674)
The function `setup_training()` tries to import `from models import *` BEFORE the Stage 2 patching logic runs. The volume's `models.py` may have broken syntax from previous failed patches (inline comments inside `spectral_norm()` calls, orphaned `import torch.nn.functional as _F` lines, broken `F.pad` blocks with `mode='reflect'` that fail on certain input sizes).

The `setup_training` pre-fix at line 648-660 only removes ONE specific broken inline comment pattern. It needs to be more robust — or better yet, it should run the FULL models.py patching logic (currently only in `train_stage2()` at lines 963-1026) BEFORE the import check.

### Problem 2: Fragile F.pad removal regex (lines 995-1019)
The regex in `train_stage2()` that removes old `F.pad` blocks from models.py is fragile:
- It uses `_re.sub()` with DOTALL flag to match multi-line blocks, but the pattern doesn't account for variations in whitespace or formatting
- It only removes blocks with exact text patterns (`"Safety: pad small inputs"`)
- The `orchard import` cleanup (line 1019) removes `import torch.nn.functional as _F\n        _min_dim = 80\n` but only if both lines appear together exactly

A robust fix would:
1. In `setup_training()`: Run the SAME full models.py patching logic (Conv2d fix + F.pad removal) BEFORE the import check, not just the inline comment fix
2. Make the F.pad removal more comprehensive — remove ALL instances of the `_F.pad` pattern, the `import torch.nn.functional as _F`, and the `_min_dim` variable, not rely on exact string matching

**IMPORTANT: After fixing, bump the `_REBUILD_TRIGGER` string (search for it in the file) from the current value to a new value like `"2026-05-02-v10"` to force Modal to rebuild the image with the updated code.**

## Files
- **Main file to fix:** `/home/ubuntu/kokoro-tamil/scripts/modal_train.py` (1238 lines)
- **Reference file (original models.py):** `/tmp/semidark-stt2/models.py` — shows what the UNPATCHED code looks like
- **Config file:** `/home/ubuntu/kokoro-tamil/configs/config_tamil_ft.yml` — already has `min_len: 192`

## Specific Fix Plan

### Fix A: Extract models.py patching into a helper function
Create a function `patch_models_py(models_path)` that:
1. Reads models.py
2. Removes ALL previous broken patches:
   - Remove any `import torch.nn.functional as _F` lines (not just when paired with `_min_dim`)
   - Remove any lines containing `_min_dim = 80` (or `\d+`)
   - Remove ALL "Safety: pad small inputs" comment blocks AND the code between them (the `if/else` block with `_pad_h`, `_pad_w`, `_F.pad`)
   - Remove any inline comments inside `spectral_norm()` that are NOT valid Python
3. Applies the Conv2d fix: `nn.Conv2d(dim_out, dim_out, 5, 1, 0)` → `nn.Conv2d(dim_out, dim_out, 5, 1, 2)`
4. Writes the clean version back
5. Returns True if patches were applied

### Fix B: Call `patch_models_py()` from BOTH `setup_training()` and `train_stage2()`
In `setup_training()` (around line 648), replace the current limited pre-fix with a call to the full `patch_models_py()`. This ensures the import check at line 662-674 sees a clean models.py.

### Fix C: Make slmadv.py patching a function too
Extract the slmadv.py patching (lines 1028-1048) into `patch_slmadv_py(slmadv_path)` and call it from both `setup_training()` and `train_stage2()`.

### Fix D: Bump rebuild trigger
Change `_REBUILD_TRIGGER` to `"2026-05-02-v10"`.

## Verification
After making changes:
1. Run `python3 -c "import ast; ast.parse(open('scripts/modal_train.py').read()); print('Syntax OK')"` to verify syntax
2. Run `git diff scripts/modal_train.py | head -100` to see changes
3. Commit and push to git

DO NOT change any other logic. Only fix the patching robustness issue.