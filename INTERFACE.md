# Tool interface contract

The local tools — **deblur** (`local_infer.py`, this repository) and **depth**
(`local_depth.py`, planned, in the DepthAnything-AC fork) — are separate programs with
separate licences, separate environments and separate models. Anything that drives both
(the parked Affinity handler, or a script of your own) is written against this contract
and nothing else. That is what "interface only, not the programs" means in practice, and
it is why this file exists rather than a shared library.

**The handler is not allowed to** import either tool's modules, parse its banner, guess
an output path, or know which model produced what. It runs a command, reads a manifest,
and works from what the manifest says.

## 1. Invocation

```
<tool> -i <file-or-folder> [--manifest <path.json>] [contract options…]
```

- `-i` takes one image or a folder of them; nothing else is required. Every default has
  to be right for an ordinary user, because the handler will not pass anything beyond
  `-i` and `--manifest` unless it has a reason.
- The same destination options on both tools: `--dest`, `--subfolder`, `--suffix`,
  `--output`, `--recursive`.
- `--manifest <path>` writes the machine-readable run record in §5. It is optional for
  humans and required for the handler.
- `--version` prints `<tool-id> <version>` on one line, and the same two strings appear
  in the manifest. `--help` must work without a GPU, a model, or an input path.

## 2. Where results go

Naming is fixed so a result can never be mistaken for an original, and so two tools do
not collide:

| Tool id | Default subfolder | Suffix | `IMG_1676.tiff` becomes |
|---|---|---|---|
| `ifan-deblur` | `deblurred` | `_db` | `…\deblurred\IMG_1676_db.tiff` |
| `depthac-depth` | `depth` | `_depth` | `…\depth\IMG_1676_depth.tiff` |

- Input names and subfolder structure are preserved apart from the suffix.
- A tool never reads its own output back as input: anything inside its result subfolder,
  or already carrying its suffix, is skipped and counted.
- The exact output path is the one in the manifest. The handler does not construct paths.

## 3. Geometry — the shared frame

This is the clause the compositing depends on:

- **A result is never resampled to make a size work.** Pixels keep the input's scale and
  their position in the frame; no tool may scale a result, and none edits the frame.
- **Deblur: the top-left crop to the largest multiple of 8.** That network downsamples by
  8, so a 4571×4578 input yields a 4568×4576 result. The crop is at most 7 px per axis
  and always takes from the bottom and right, so the top-left of the composition is
  exact.
- **Depth: no crop at all.** Its network works on a resampled frame of its own (518 px
  short side, §7) and interpolates the map back to the input's full size, so a depth map
  is exactly as large as its input — which is what a mask for the document needs.
- The two therefore differ by at most 7 px per axis for the same photo. Both begin at the
  input's top-left, both report `[0, 0]`, and a handler places them there; it never
  resolves the difference by resampling, because nothing here may be resampled.
- The manifest carries both sizes and the offset (§5), so the handler is told rather
  than left to assume.

How a tool reports this for a human is its own business; how it reports it to the
handler is not.

## 4. Bit depth and channels

- **Deblur** follows the input file: 16-bit in gives 16-bit out, 8-bit in gives 8-bit
  out, no arguments, per file, in a mixed folder. The 16-bit container is TIFF or PNG.
- **Depth** writes 16-bit greyscale, always: depth is a smooth gradient used at partial
  opacity, which is where 8 bits bands. The 8-bit colour map it writes beside the result
  is a preview for the eye, not a depth map, and is never the `output`.
- **Channels:** 3 for a deblur result, 1 for a depth result. An input alpha channel is
  not carried through to a deblur result — the original still has it, and a correction
  layer wants the whole frame. That is a condition of the output, not a warning.
- The manifest states what was actually written (`bit_depth`, `channels`, `container`),
  so the handler never infers it from the extension.

## 5. The manifest

One JSON object per run, written even when some files failed:

```json
{
  "contract": 1,
  "tool": {"id": "ifan-deblur", "version": "1.0.0"},
  "started": "2026-10-18T17:31:06+00:00",
  "finished": "2026-10-18T17:31:35+00:00",
  "device": "cuda (NVIDIA GeForce RTX 3060)",
  "options": {"dest": "both", "suffix": "_db", "tile": 1024, "overlap": 256,
              "max_pixels": 158000000, "config": "auto", "ext": "keep"},
  "counts": {"found": 4, "written": 4, "failed": 0, "skipped": 0},
  "results": [
    {
      "input": "D:\\photos\\IMG_1676.tiff",
      "output": "D:\\photos\\deblurred\\IMG_1676_db.tiff",
      "status": "ok",
      "error": null,
      "input_size": [4571, 4578],
      "output_size": [4568, 4576],
      "offset": [0, 0],
      "bit_depth": 16,
      "channels": 3,
      "container": "tiff",
      "seconds": 7.26
    }
  ]
}
```

- `contract` is the version of *this* document. A handler that does not recognise the
  number must refuse the run rather than guess.
- `status` is `ok`, `failed` or `skipped`. A `failed` entry carries the same message the
  console showed in `error`, and `output` is `null`.
- Sizes are `[width, height]`, in the file's own orientation.
- `bit_depth` is per channel. `container` is `tiff`, `png` or `jpeg`. A JPEG is written
  with the input's own chroma sampling — 4:4:4, 4:2:2 or 4:2:0 as it arrived, and 4:4:4
  from a lossless input — so the container does not quietly throw away colour resolution
  the input had.
- `color_map` is optional and may be `null`. Depth writes an 8-bit colour preview beside
  its map and names it here; a handler may ignore it. It is never a substitute for
  `output`, and the preview carries no information the 16-bit map does not.
- Console output stays exactly as it is — one line per file, readable — because a human
  reads it. The manifest exists so the handler does not have to.

## 6. Failure and exit

- One bad file never stops a batch: it is reported, counted, and the run continues.
  This is why the tools are separate processes in the first place.
- Exit code `0` when every requested file was written; non-zero when any failed. A
  handler that sees `0` may trust the manifest; one that sees non-zero reads it anyway,
  because partial runs are normal and the manifest says which parts.
- A refusal (past the memory ceiling, unreadable container, bit-depth mismatch) is a
  `failed` entry with a message naming the limit and the fix. Refusals are not silent.

## 7. What this contract does not cover

- **Blending.** How much of a correction layer to use is the artist's decision by eye,
  taken in Affinity's own masking and opacity. No tool here composites, and the handler
  does not either.
- **Quality settings and model choice.** `--tile`, `--max-pixels` and friends are the
  tool's business; the handler passes them through only if it was asked to.
- **The depth model's working resolution.** Depth Anything works at a fixed short side
  and interpolates the map back up; that is the model's nature, and it is reported on
  the console and in the manifest, not hidden. A tiled native-resolution pass is a
  possible later change, not part of this contract.

## 8. Status

| Clause | `ifan-deblur` | `depthac-depth` |
|---|---|---|
| 1 Invocation, `-i`, destination options | **yes** | **yes** |
| 1 `--version` | **yes** | **yes** |
| 2 Naming and re-ingest guards | **yes** | **yes** |
| 3 Shared frame, no resampling | **yes** — /8 crop | **yes** — no crop |
| 4 Bit depth and channels | **yes** — follows the file | **yes** — 16-bit grey, 1 channel |
| 5 Manifest | **yes** | **yes** — plus `color_map` |
| 6 Per-file failure, exit code | **yes** | **yes** |

Both sides of the table are now real, with the gaps of §8 closed: `--version` needs no
checkpoint and no image, and the manifest is written as the run proceeds — the header
before the first image, then rewritten with each result as it lands, through a temporary
file and an atomic replace, so an interrupted run still leaves a record of everything it
had finished. Each has been checked for an ordinary batch, for a per-file failure (a
`failed` entry carrying the console's message, exit code 1) and for a re-run that skips
a tool's own output.

The two tools work on the same photo at the same time without sharing a line of code:
`local_infer.py` in the IFAN tree, `local_depth.py` in the depth tree, each with its own
licence and its own dependencies. The handler is the next piece, and it talks to them
through this document.

## The licence reason this is a contract and not a library

`ifan-deblur` is AGPL-3.0. `depthac-depth` is CC-BY-NC-4.0 — non-commercial — because
it carries the Depth Anything AC model and Meta's DINOv2 backbone, and it also contains
Apache-2.0 code from Depth Anything V2. Code from those cannot be merged into one
program without mixing terms that do not mix. Two processes and a documented contract
share nothing: the handler can carry whatever licence its author chooses, and neither
tool constrains the other.

_Last changed 2026-10-18._
