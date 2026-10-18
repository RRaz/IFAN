# IFAN — Local Setup and Usage

Instructions for running this checkout of [IFAN](https://github.com/RRaz/IFAN)
(Iterative Filter Adaptive Network for single-image defocus deblurring, CVPR 2021)
locally on Windows with a CUDA GPU.

This file documents **this working copy** and the changes made to it. The
upstream project documentation is in [`README.md`](README.md) and is unchanged.

---

## 1. What this copy adds

| File | Status | Purpose |
|---|---|---|
| `predict.py` | **modified** | `max_side` raised `1920` → `5760`. Only affects the Cog/Replicate path, which is not used locally |
| `local_infer.py` | **new** | Batch runner: processes folders of images, tiles large ones, follows each file's bit depth |
| `run_infer.bat` | **new** | Windows wrapper around `local_infer.py` (drop zone + drag-and-drop) |
| `demo/input/input_large.jpg` | **new** | 4532x4578 (20.7 Mpx) sample that exercises tiling |
| `input/` | **new** | Drop-zone folder used by `run_infer.bat` |
| `output/` | created on demand | Destination only for `--dest output`; the default writes beside the inputs, so it normally does not exist |
| `.gitignore` | **new** | Excludes `.venv/`, `ckpt/`, `deblurred/`, `*_db.*`, `output/`, Python caches |
| `ckpt/IFAN.pytorch` | **new** | Pretrained weights, 113 MB, 8-bit (not for committing) |
| `ckpt/IFAN_16bit.pytorch` | **new** | Pretrained weights, 113 MB, 16-bit (§5.6; also not for committing) |
| `.venv/` | **new** | Python 3.11 virtual environment |

Everything else is unmodified upstream code.

---

## 2. Size limits, and how this runner handles them

Upstream IFAN shrinks any input whose longest side exceeds **1920 px** before
deblurring (`predict.py: max_side = 1920`), so large images come back *smaller
than they went in*.

Raising that number alone is not enough, because memory is the real constraint.
IFAN's filter predictor (`models/archs/IFAN.py`, `self.F`) emits a
**15,232-channel** tensor at 1/8 resolution, so whole-image inference costs roughly
**1.8 GB per megapixel**:

| Input | Whole image | Tiled |
|---|---|---|
| 2400x1600 (3.8 Mpx) | 6793 MiB | ~1884 MiB |
| 3000x2000 (6.0 Mpx) | 10592 MiB | ~1884 MiB |
| 5760x3840 (22.1 Mpx) | 19.6 GiB in one allocation — fails on a 12 GB card | ~1884 MiB |

`local_infer.py` therefore does three things, none of which need arguments:

1. **Never downscales.** Full resolution is the default.
2. **Tiles automatically.** Anything larger than `--tile` (1024 px by default) is
   split into overlapping tiles and feathered back together, so peak GPU memory
   stays flat at ~1.9 GB however big the input is (§6).
3. **Stops at a ceiling instead of shrinking.** An image over **100 megapixels** is
   refused with a message that names the limit, rather than silently coming back
   smaller:

   ```
   [  1/3] FAILED big.jpg: 3000x2000 = 6.0 megapixels is over the 1.0 megapixel limit
                           (raise it with --max-pixels if you have the RAM and time)
   ```

   The ceiling exists only to stop a runaway job that would exhaust host RAM, and
   it is far above any ordinary photo. Change it with `--max-pixels`, or pass
   `--max-pixels 0` for no ceiling at all.

A batch keeps going when one image is refused — the others are still processed (§5.5).

---

## 3. Requirements

- Windows with an NVIDIA GPU and a recent driver (developed against an RTX 3060 12 GB, driver 610.88)
- [uv](https://docs.astral.sh/uv/) on `PATH`
- ~4 GB free disk for the environment plus 113 MB for the weights

The system-wide Python is 3.14, which this project does **not** use — everything
runs inside `.venv` (Python 3.11).

Installed versions:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.14.1+cu132 |
| torchvision | 0.29.1+cu132 |
| numpy | 2.4.6 |
| opencv-python | 5.0.0.93 |
| easydict | 1.13 |
| pillow | 13.0.0 |
| rawpy | 0.27.1 (libraw 0.22.1) — optional; not used by the runner, see §8 |

---

## 4. Setup from scratch

Only needed if `.venv` or `ckpt/` are missing.

### 4.1 Environment

Run from the project root (the folder containing `local_infer.py`):

```bat
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend=auto torch torchvision
uv pip install --python .venv\Scripts\python.exe numpy opencv-python easydict tqdm termcolor
```

`--torch-backend=auto` picks the correct CUDA build for your driver. The venv must
live at `.venv` in the project root — `run_infer.bat` looks for it there.

### 4.2 Pretrained weights

Download `checkpoints.zip` (916 MB) using either link from `README.md`:

```bat
curl.exe -L --fail -o checkpoints.zip "https://www.dropbox.com/s/qohhmr9p81u0syi/checkpoints.zip?dl=1"
```

The archive extracts **flat** (no `ckpt/` wrapper). Keep `IFAN.pytorch`:

```bat
mkdir ckpt
Expand-Archive checkpoints.zip -DestinationPath ckpt_tmp
copy ckpt_tmp\IFAN.pytorch ckpt\IFAN.pytorch
```

The archive also contains `IFAN_8`, `IFAN_26`, `IFAN_35`, `IFAN_44`, `IFAN_dual`
and `IFAN_16bit`. Each needs its matching config (`config_IFAN_8`, etc.), which the
repo already provides, but `local_infer.py` only knows the two that matter here:

```bat
copy ckpt_tmp\IFAN_16bit.pytorch ckpt\IFAN_16bit.pytorch   REM the 16-bit weights; see §5.6
```

Keep `IFAN.pytorch` (8-bit) and `IFAN_16bit.pytorch` (16-bit). Delete the rest if you do
not need them. `IFAN.pytorch` is the one that has to be there: without
`IFAN_16bit.pytorch` a 16-bit input still works and still comes back 16-bit, but the
correction applied is the 8-bit model's and the run says which weights it used.

---

## 5. Usage

### 5.1 Batch file (recommended)

There are two ways to use it, and neither needs any arguments.

**Drop zone.** Put images, or folders of images, into the `input` folder next to
`run_infer.bat`, then double-click the file.

**Drag and drop.** Drag folders (or individual images) onto `run_infer.bat` in
Explorer. Several items can be dropped at once.

Either way, **results land beside each input**, in a `deblurred` subfolder and with
a `_db` name suffix, so they can never be mixed up with the originals:

```
D:\photos\IMG_0001.jpg    ->   D:\photos\deblurred\IMG_0001_db.jpg
input\holiday\a.png       ->   input\holiday\deblurred\a_db.png
```

Explicit paths also work — any argument that exists on disk counts as an input:

```bat
run_infer.bat "D:\photos" "E:\more photos"
run_infer.bat "D:\photos" --dest output        REM into the project output folder
run_infer.bat "D:\photos" --tile 512           REM less GPU memory
```

No options are needed for normal use: the defaults never downscale, tile large
images automatically, keep each file at its own bit depth (§5.6), and stop with a
message if an image is past the ceiling. §5.4 lists the other destination layouts.

When the file is double-clicked the window stays open at the end so the results
are readable. Set `IFAN_NO_PAUSE=1` to suppress that, e.g. when calling it from
another script.

The wrapper always passes `--recursive`, so subfolders are always processed. For
a flat, non-recursive run, call `local_infer.py` directly.

### 5.2 Directly

```bat
.venv\Scripts\python.exe local_infer.py -i <input> [options]
```

`<input>` may be a single image or a folder. Results are written next to it unless
a destination is named.

### 5.3 Options

| Option | Default | Meaning |
|---|---|---|
| `--input`, `-i` | *required* | Input image or folder |
| `--dest` | `both` | Where results go: `both`, `subfolder`, `suffix` or `output` (see §5.4) |
| `--subfolder` | `deblurred` | Subfolder name used by `--dest subfolder` and `both` |
| `--suffix` | `_db` | File-name suffix used by `--dest suffix` and `both` |
| `--output`, `-o` | `<project>\output` | Destination root for `--dest output`; naming it implies `--dest output` |
| `--ckpt` | per bit depth | Checkpoint to load; by default the matching one, falling back to `ckpt\IFAN.pytorch` when it is absent |
| `--max-side` | `0` | Downscale inputs whose longest side exceeds this. **`0` = never downscale** |
| `--max-pixels` | `100000000` | Refuse images above this many pixels rather than shrinking them. **`0` = no ceiling** |
| `--tile` | `1024` | Largest tile side. **`0` disables tiling** (memory then scales with the image) |
| `--overlap` | `256` | Feather width between neighbouring tiles |
| `--recursive`, `-r` | off | Recurse into subfolders, preserving the folder structure |
| `--cpu` | off | Force CPU even when a GPU is present |
| `--config` | `auto` | `auto` follows each file's bit depth (§5.6); `config_IFAN` forces 8-bit, `config_IFAN_16bit` forces 16-bit |
| `--ext` | `keep` | Output format: `keep`, `png`, `jpg` or `tif` |

`--tile` and `--overlap` are rounded down to a multiple of 8 (the network
downsamples by 8), and `--overlap` is capped at half the tile size.

### 5.4 Where results go

`--dest` picks one of four layouts. The default is `both`, since a result that is
both in its own folder and named as a result cannot be mistaken for an original.

| `--dest` | `D:\photos\IMG_0001.jpg` becomes | Notes |
|---|---|---|
| `both` *(default)* | `D:\photos\deblurred\IMG_0001_db.jpg` | Subfolder **and** suffix |
| `subfolder` | `D:\photos\deblurred\IMG_0001.jpg` | Suffix off |
| `suffix` | `D:\photos\IMG_0001_db.jpg` | Beside the original, no new folder |
| `output` | `<--output>\IMG_0001.jpg` | Separate root, default `<project>\output` |

Whichever you pick:

- **File names and extensions are preserved** apart from the suffix.
- **Subfolders are mirrored** inside the destination, rooted at the input folder.
  `D:\photos\evening\a.jpg` becomes `D:\photos\deblurred\evening\a_db.jpg`.
- **Results are never re-processed as inputs.** Everything under `deblurred`, and
  anything already ending in `_db`, is skipped and counted in a `skipped :` line.
  Re-running therefore replaces results instead of piling up `_db_db` copies.
- **Existing outputs are overwritten**, so a re-run replaces the previous results —
  it does not skip work that is already done.
- Outputs are written in BGR via OpenCV, so colours match the input.
- Dimensions are trimmed to a multiple of 8 — a **1–7 px** reduction, inherent to
  the network's /8 downsampling. This is not the downscaling described in §2.

### 5.5 More than one image in the folder

Every image is processed in one pass, one at a time:

```
images     : 3 found under D:\photos
[  1/3] big.jpg  3000x2000 -> 3000x2000 [12 tiles]  2.51s peak GPU 1884 MiB
[  2/3] medium.jpg  1200x800 -> 1200x800 [2 tiles]  0.31s peak GPU 1481 MiB
[  3/3] small.jpg  320x240 -> 320x240  0.02s peak GPU 175 MiB
Done: 3 of 3 image(s) in 2.8s (average 0.95s)
```

- **Files, not folders, are the unit of work.** Subfolders are walked when
  `--recursive` is on (the batch file always sets it) and the structure is mirrored
  into the destination folder.
- **Checkpoints are loaded once each, on demand.** At most one load per configuration in
  a run, so a folder of 200 photos does not pay the ~1 s load 200 times, and a folder of
  mixed 8-bit and 16-bit files pays it twice and no more (§5.6).
- **Bit depth is decided per file**, not per run, so a folder holding 8-bit JPEGs and
  16-bit TIFFs together comes out right in a single pass (§5.6).
- **Each image is tiled independently**, so one huge image mixed in with small ones
  does not increase peak memory for the rest of the batch.
- **A failure skips that image and continues.** A corrupt file, an unreadable
  format, or one past the pixel ceiling is reported and the batch carries on. The
  last line counts both outcomes:
  `Done: 2 of 3 image(s) in 0.5s (average 0.24s), 1 failed`.
- **Results are never fed back in as inputs**, so a second run over the same folder
  cannot grow it (§5.4). It does redo the work, overwriting the earlier results.
- Order is deterministic (sorted by path), so a re-run processes the same sequence.

### 5.6 16-bit (48-bit colour) output

**Bit depth follows the file, so there is nothing to set.** A 16-bit TIFF or PNG is read,
deblurred and written at 16 bits, an 8-bit JPEG stays 8-bit, and a folder holding both gets
the right treatment for each — the two models are loaded on demand within the one run. 16
bits each in red, green and blue is Affinity's 16-bit mode. Each result line names the depth
it was written at:

```
[  2/2] IMG_1676_16.tiff  4571x4578 -> 4568x4576  16-bit [36 tiles]  7.22s peak GPU 1924 MiB
```

`--config` overrides the automatic choice when one mode is wanted for a whole run — here
forcing 8-bit output whatever the inputs are:

```bat
.venv\Scripts\python.exe local_infer.py -i D:\photos --config config_IFAN --ext png
```

Three things have to line up, and every one of them is checked rather than assumed:

- **The weights for that depth.** 16-bit input wants `ckpt\IFAN_16bit.pytorch` (§4.2).
  When it is not there the run falls back to the 8-bit checkpoint rather than refusing,
  because the two architectures are identical and either loads into either, and it says
  which one it used:

  ```
  note       : IFAN_16bit.pytorch is not there, so the 8-bit checkpoint is used for
                config_IFAN_16bit: the correction is the 8-bit model's, carried through
                at full precision.
  ```

- **The container for that depth.** JPEG cannot hold 16 bits and neither can WebP, so a
  16-bit result needs TIFF or PNG — and since a 16-bit input already is one, `--ext keep`
  gets this right on its own. `--ext jpg` on a 16-bit file is a real contradiction and is
  refused, per file, with the reason. A DNG can be read but not written back out, so a
  DNG input produces a TIFF: leaving the user to discover which flag fixes that would be
  worse than choosing the only container that works.

- **The config against the file, in both directions.** `config_IFAN_16bit` divides by
  65535, so forcing it on an 8-bit file would come back almost black; that is refused
  instead of being written out. Forcing `config_IFAN` on a 16-bit file says what it is
  throwing away:

  ```
  WARNING    : IMG_1676_16.tiff holds 16-bit data but config_IFAN reads and writes 8-bit,
                so the extra precision is discarded; drop --config to keep each file at
                its own depth
  ```

- **Alpha.** Affinity's TIFF exports often carry an alpha channel. `IMREAD_UNCHANGED`
  returns it as a fourth channel, and upstream's 16-bit read would reorder it into the
  colour channels and corrupt the picture, so this runner drops alpha first — and says so,
  because the result then has no transparency of its own.

**16-bit input and output do not by themselves make the model deblur better.** What they
buy is the precision to hand the result to a blend — which is the requirement, and is also
why the 16-bit-trained weights matter less here than they look: nothing below depends on
them.

Blending sets that requirement, because a blend scales the layer's contribution down.
Measured on a real 20.9-megapixel frame, compositing the deblurred result over the
original and comparing against the same blend computed exactly in floating point:

| layer opacity | levels the correction can still take, 8-bit layer | 16-bit layer |
| --- | --- | --- |
| 100% | 295 | 42,015 |
| 50% | 152 | 22,797 |
| 30% | 94 | 14,932 |
| 10% | 33 | 5,706 |
| 5% | 17 | 3,076 |

At the subtler opacities an artist actually uses, an 8-bit layer has only tens of levels
left for the thing being blended in, so a smooth correction arrives stepped; a 16-bit
layer has thousands. The effect is not confined to where the layer bites — at 30% the
8-bit composite differs from the exact one by 0.32 of a level at the median pixel and
3.4 at the worst, with 1.9% of pixels moving by more than a level, whereas at 16 bits
the same comparison is 0.001 of a level. A stack is only as deep as its shallowest
layer, so one 8-bit layer in the composite quantises all of it.

Bit depth does **not** change GPU memory: the network works in `float32` either way and
the two configs are architecturally the same, so the ~1.9 GB tiled peak in §6 applies
unchanged.

It does change the size on disk, and there is little to be won back. At 20.7 megapixels
a 16-bit result is **~108 MiB** — 3 channels x 2 bytes x 20.7 Mpx is 119 MiB of pixels to
start with, and because the low byte of a smooth 16-bit gradient is close to random,
lossless compression only just beats that. (16-bit PNG comes out the same 108 MiB.)
`local_infer.py` writes 16-bit TIFF with Adobe Deflate for this reason; OpenCV's LZW
default would *inflate* the same picture to 147 MiB, larger than raw. 8-bit output is
unaffected. If you need a 16-bit file small enough to move around, **crop it** — scaling
it down is what throws the detail away.

---

## 6. How tiling works

1. The image is split into tiles of `--tile` px, stepping by `tile - overlap`.
2. Tile origins are kept multiples of 8 so features line up across tiles, and the
   final tile in each row/column ends flush with the image edge.
3. Each tile is run through the network independently.
4. Results are feathered with a smoothstep window and combined as a weighted
   average. Mirrored ramps sum to exactly 1, so blending cannot amplify noise at
   a seam. Outer image edges get weight 1 since there is no neighbour to blend with.

Measured on an RTX 3060 (12 GB), tiled vs. whole-image processing:

| Image | Whole-image | Tiled | Tiles |
|---|---|---|---|
| 2400x1600 | 6793 MiB, 0.84 s | **1884 MiB**, 1.60 s | 6 |
| 3000x2000 | 10592 MiB, 2.85 s | **1884 MiB**, 2.52 s | 12 |
| 5760x3840 | **out of memory** | **1884 MiB**, 8.06 s | 40 |

Accuracy against the whole-image result (ground truth): **58.65 dB** and
**57.46 dB** PSNR, mean absolute difference 0.07–0.08 of 255, worst-case
difference 6 of 255, with 0.21–0.28% of pixels differing by more than 2 levels.

Tuning: smaller `--tile` means less memory, more seams and slower runs. Peak
memory scales roughly with the square of the tile size, so halving `--tile` to
512 reduces memory to about a quarter of the measured 1884 MiB.

---

## 7. Verifying

`demo\input\` holds two samples: `input.jpg` (640x425, too small to tile) and
`input_large.jpg` (4532x4578, **20.7 megapixels**). Run the large one and read the
console:

```bat
run_infer.bat demo\input\input_large.jpg
```

```
device     : cuda (NVIDIA GeForce RTX 3060)
max side   : unlimited
tiling     : 1024 px tiles, 256 px overlap
pixel limit: 100 megapixels
images     : 1 found under demo\input\input_large.jpg
destination: demo\input\deblurred as <name>_db<ext>
[  1/1] input_large.jpg  4532x4578 -> 4528x4576 [36 tiles]  7.28s peak GPU 1884 MiB
Done: 1 of 1 image(s) in 7.3s (average 7.28s)
```

Two things to check:

- `->` dimensions match the input, minus up to 7 px of /8 trimming — so **nothing was
  downscaled**. This is the behaviour upstream got wrong.
- `peak GPU` stays near ~1900 MiB even as the image grows, and `[N tiles]` is
  reported — so **tiling is engaging**. A climb into the thousands means it is not,
  and the image was processed whole.

The result lands at `demo\input\deblurred\input_large_db.jpg`, beside the original.

To see the ceiling work, push it down and run any ordinary photo:

```bat
.venv\Scripts\python.exe local_infer.py -i "D:\photo.jpg" -o out_test --max-pixels 1000000
```

```
[  1/1] FAILED photo.jpg: 3000x2000 = 6.0 megapixels is over the 1.0 megapixel limit
                          (raise it with --max-pixels if you have the RAM and time)
Done: 0 of 1 image(s) in 0.0s, 1 failed
```

Tiling introduces no visible seams. On `input_large.jpg`, comparing tile 1024
against tile 512 gives **46.5 dB PSNR**, with only 0.02% of pixels differing by more
than 8 levels — and those differences are *less* likely to fall on a tile boundary
than anywhere else. Measuring gradient strength directly across the tile lines shows
no elevation either. If you do ever see faint straight lines, raise `--overlap`
(e.g. `384`) and re-check.

The 16-bit path is verified the same way, and needs no options either. Build a 16-bit
copy of an 8-bit image by scaling every channel by exactly **257** (since 257 × 255 =
65535): the 16-bit result, divided by 257, must then be **identical** to the 8-bit result
for the same image, because both normalise to the same values in [0, 1]. Anything else
means the read or write scaling is wrong. Including an alpha channel in that test file
must not change the result by a single byte.

Two things are worth re-checking whenever `local_infer.py` changes:

- **The 8-bit path is byte-identical.** Its output is what upstream users already had, so
  it is compared as a file rather than by eye. Run the same 8-bit image through the
  previous revision and the new one and compare the results with
  `Get-FileHash <a>,<b> -Algorithm SHA256`. An 8-bit JPEG deblurred to PNG on
  `demo/input/input_large.jpg` must keep coming out as `0894BB42 6EB24483 BECEB3B3
  004B61BF 0653500D FDEF3FCC 3769BDA3 D7074F09`.
- **Bit depth follows the file.** A folder holding one 8-bit JPEG and one 16-bit TIFF must
  report `8-bit` on one line and `16-bit` on the other, loading `ckpt\IFAN.pytorch` for the
  first and `ckpt\IFAN_16bit.pytorch` for the second, and leaving a `uint8` and a `uint16`
  file on disk.

---

## 8. Troubleshooting

**`CUDA out of memory`**
Tiling is not engaging, or `--tile 0` was passed. Lower `--tile` (e.g. `512`).
Tiled runs need ~1.9 GB regardless of input size.

**`... is over the N megapixel limit`**
Not the network failing — it is the guard from §2. Either shrink the image, raise
the ceiling (`--max-pixels 200000000`), or remove it (`--max-pixels 0`). Each extra
100 megapixels needs roughly 2 GB of host RAM and about 40 s on an RTX 3060.

**`No checkpoint found in ...\ckpt`**
Neither `IFAN.pytorch` nor `IFAN_16bit.pytorch` is present — see §4.2. A
`Checkpoint not found: <path>` on its own means `--ckpt` named a file that is not there.

**`Python environment not found`**
`.venv` is missing or was moved away from the project root — see §4.1.

**Colours look inverted**
Outputs are converted from the network's RGB to OpenCV's BGR before writing. If
you post-process the outputs with a tool that assumes RGB, account for that.

**`predict.py` fails with a missing-argument error**
It is broken upstream and unusable locally as committed:

```python
ckpt_manager = CKPT_Manager(root_dir='', model_name='IFAN')   # `cuda` argument is required
```

It also requires the `cog` package, which targets Linux/Replicate deployments.
This is why `local_infer.py` exists. The `max_side = 5760` edit is still applied
in `predict.py` for anyone deploying through Cog.

**`eval.py` fails with `unexpected keyword argument 'multichannel'`**
`eval.py` calls `structural_similarity(..., multichannel=True)`, which modern
`scikit-image` removed in favour of `channel_axis`. `local_infer.py` does not use
`eval.py`, so this only matters if you run the upstream evaluation path, which
additionally needs `scikit-image`, `scikit-learn`, `scipy`, `tensorboardX`,
`ptflops` and `torch-optimizer`.

**`A DNG is ignored, or does not match the same picture as a TIFF`**
`.dng` is deliberately absent from the extensions a folder scan collects, so a batch over
a photo folder will not touch raw files at all — a run that seems to prove something
about DNG has probably never opened one. Name it directly (`-i photo.dng`) to try it. Even
then OpenCV sees only the TIFF container: the colour matrix, white balance and
linearisation that make a DNG a DNG are skipped, and the runner says so on the console. A
camera raw usually arrives as a single-channel colour-filter array rather than a picture,
which it also warns about. A DNG can be read but not written back out, so a DNG input
produces a **TIFF** and the run says so, rather than stopping to ask for a flag it could
have chosen itself. Its depth is probed like any other file, so a 16-bit raw comes back
16-bit. Affinity can read a DNG but cannot write one, so a DNG is not
available as an interchange format for this workflow in any case.
For a correct read `rawpy` (libraw) is installed in `.venv`, but for anything going into
the deblurring pipeline, export a demosaiced TIFF or PNG instead.

**Input is very small**
Images under 8 px after trimming are rejected, since the network needs at least
one 8x8 feature-map cell.

---

## 9. Notes for contributors

- `ckpt/` holds 113 MB of weights and is listed in `.gitignore`. Keep it out of commits.
- `.venv/` is ignored only because `uv` writes a `.gitignore` containing `*` inside
  it — upstream ships no root `.gitignore`, so the one here was added for
  `.venv/`, `ckpt/`, `deblurred/`, `*_db.*`, `output/` and Python caches.
- `input/*` is ignored except `input/README.txt`, so the drop zone stays clean in `git status`.
- The only tracked change is `predict.py`; `local_infer.py`, `run_infer.bat`,
  `LOCAL_SETUP.md` and `.gitignore` are new.
- The published checkpoint's keys are prefixed `module.Network.` and the file also
  contains a training-only reblurring network. `local_infer.py` detects the prefix
  automatically and loads only the 158 parameters the inference network owns,
  refusing to run if nothing matches.
- Keep this working copy outside OneDrive or any other file-sync folder. Sync
  services interfere with multi-gigabyte environments (hardlinking fails, files
  get locked mid-run). This copy lives at `D:\Projects\IFAN`.
