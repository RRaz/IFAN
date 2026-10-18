# Driving IFAN from Affinity 3 — open question

**Status:** add-on, not started, and deliberately not part of the current work · captured 2026-10-18

**The interface it will be written against is now fixed:** [`INTERFACE.md`](INTERFACE.md).
The deblur tool satisfies it already except for the manifest and `--version`; the depth
tool is to be brought up to it; the handler comes last and drives both through it.

This is kept separate on purpose. The 16-bit path (`LOCAL_SETUP.md` §5.6) is finished and
tested on its own first, so that the two are not being debugged at the same time. Nothing
here blocks anything there, and nothing there needs reading in order to use this runner.

An idea to pick up later: can Affinity 3 (by Canva) hand an image to this project
and take the deblurred result back — as a *layer*, not a flattened file?

## Why it is worth doing

Photographers already working in Affinity 3 would get deblurring as a step inside
their normal edit. The workflow this repo is built for is mask-and-blend **by eye**:
the deblurred plate is composited back over the original and a mask decides where
it applies, so sky and distant objects keep the softer rendering they should have.
Uniform deblurring across the frame fights how a photograph is read.

Because the plate is normally blended at partial opacity rather than at 100%, the
precision of the stack matters more than the precision of the deblur itself — a blend
scales the layer's contribution down, and an 8-bit layer has little left to give at that
point. That is what makes 16-bit worth having; see the bit depth section below.

That fixes the shape of any integration: it must return a **layer of the same
dimensions**, so it drops onto the stack at 100% with exact pixel alignment and no
resampling. Anything that returns only a flattened file is the wrong shape for the
job.

## What is confirmed

**Two interfaces exist.**

1. **JavaScript scripting** (Affinity 3)
   - Disabled by default; enable it in Affinity's Settings.
   - API reference: <https://affin.co/affinity-sdk>, which lives outside the Help
     Centre. Documents modules and classes such as `Document`, `Dialog` and
     `Collection`.
   - The Affinity UI can only *run* scripts — you cannot view, edit or create them
     inside Affinity. Installed scripts appear in the Scripts panel
     (`Window > General > Scripts`).

2. **Local MCP server** (3.2 and later; the official guide describes it as beta)
   - Listens on `http://localhost:6767/sse`.
   - Off by default. Enable it in Affinity's Settings; Affinity restarts.
   - Claude Desktop is the named supported client.
   - Carries Affinity's own SDK documentation and example scripts for the agent to
     read, so the API surface is discoverable from the server itself.
   - Tools seen so far: `search_sdk_hints`, `add_sdk_hint`, and script-library
     management (add / list / save — no delete; delete in the Scripts panel).

**Community tooling that already works around the limitations**

- `rabidgremlin/affinity-scripting` — `extract_docs.js` dumps every SDK doc topic
  and the bundled example scripts to a local folder; `script_mgr.js` adds, lists
  and saves scripts in the Affinity library; `search_sdk.js` queries SDK hints.
- `JiriKrblich/Affinity-script-manager` — GUI for authoring `.js` scripts and
  pushing them into Affinity through the local MCP bridge.
- Reaching the server from a non-Claude client: proxy it to stdio, e.g.
  `uvx mcp-proxy --transport sse http://localhost:6767/sse`.
- Inspecting the live tool list:
  `npx @modelcontextprotocol/inspector --sse http://localhost:6767/sse`.

## The open question

Neither interface has been confirmed able to:

- **run an external process** (to invoke `local_infer.py`), or
- **export the current document to a chosen path**, and **place an imported image
  back as a layer**.

Partial evidence that file access may be restricted: the community markdown
example notes the file *must be on the Desktop for the file picker to select it*,
which points at a native file picker rather than arbitrary paths. If that is the
model, handing work off to a command-line tool is awkward.

Do **not** assume an API such as `api.runProcess` carries over from other
applications — that call belongs to Cavalry, not Affinity.

## Bit depth — 8-bit today, 48-bit wanted

Second priority, **after** the interface is working: 16 bits per channel (48-bit
colour), matched to Affinity's 16-bit mode. The findings below make this a much
smaller job than it sounds.

**Implemented on 2026-10-18** — see `LOCAL_SETUP.md` §5.6. Bit depth is now taken from each
file, with the matching weights and container chosen to suit, so a 16-bit input needs no
arguments at all; `--config` remains as an override for forcing one mode, `IFAN_16bit` is
the mode behind it, and the traps listed further down are guarded. Verified against the
8-bit result: a 16-bit file built by scaling an
8-bit image by exactly 257 yields a 16-bit result that divides back by 257 to a
**byte-identical** picture, and the same file carrying an alpha channel gives
byte-identical output as well.

How it works, for reference:

- The whole pipeline is **float32** — the network runs on float patches and the tile
  accumulator is `float32` — so nothing is lost in the middle.
- **16-bit input is already supported upstream.** `data_loader/utils.py::read_frame`
  branches on `norm_val`: when it equals `2**16-1` it reads with
  `cv2.imread(path, -1)` (`IMREAD_UNCHANGED`, which preserves 16-bit depth) and
  divides by 65535. Otherwise it reads with `IMREAD_COLOR`, which **downconverts to
  8-bit**.
- `local_infer.py` now selects the config by name, mapping each `--config` value to
  that config's own `get_config` and mode. The 8-bit path still calls `read_frame`
  unchanged, and its output is byte-for-byte what it was before the change.
- The 16-bit path reads with `IMREAD_UNCHANGED` itself, so that it can drop an alpha
  channel before the channel reversal — see the traps below.
- The write is one line: scale by `2 ** in_bit - 1`, cast to `uint16` for the 16-bit
  config and `uint8` otherwise. Same code, different divisor.
- `--ext` gained `tif` alongside `keep`, `png` and `jpg`.

What is still missing:

- **The 16-bit trained model.** `configs/config_IFAN_16bit.py` (`in_bit = 16`,
  `norm_val = 65535`, same `N = 17` and `refine_val = 8`) with
  `ckpt/IFAN_16bit.pytorch`. The two architectures are identical, so the 8-bit
  checkpoint loads into the 16-bit config — that is what has been run so far — but
  whether the model deblurs *differently* at that precision needs weights trained for
  it. They are not in this repo; they come from the same upstream download as the 8-bit
  ones ("Our model trained with 16 bit images", Table 1 of the supplementary material).

  **This turned out to be the smaller half of the question.** 16-bit is not wanted in
  order to deblur better; it is wanted so that the result survives being blended, since a
  blend scales the layer's contribution down and the stack is only as deep as its
  shallowest layer. Measured on a real 20.9-megapixel frame, an 8-bit layer leaves 33
  levels for the correction at 10% opacity and 94 at 30%, where a 16-bit layer leaves
  5,706 and 14,932. None of those numbers involve the 16-bit weights, so the part that
  matters most was working before the weights arrived.

  Both checkpoints are now local (`ckpt\IFAN.pytorch` and `ckpt\IFAN_16bit.pytorch`), so
  the remaining question — whether the 16-bit-trained model's correction differs visibly
  from the 8-bit model's — can be answered by running the same frame through both.

- **A genuine 16-bit image to try it on.** Supplied: `testfolder/IMG_1676_16.tiff`,
  4578 x 4571, holding **65,477 distinct levels** — not the 256 that an 8-bit picture
  scaled into a 16-bit container would show, so the export chain is not collapsing to 8
  bits on the way out. Its 16-bit PNG twin carries the same data to a mean difference of
  0.000, so the container choice is only about size (105.6 against 104.0 MiB — no
  meaningful difference). Affinity writes those TIFFs with Adobe Deflate
  (Compression=8), the same choice `local_infer.py` now makes, so the two are
  interchangeable as input. Whether the *source* held more than 8 bits of real
  information is a separate question that only the file's provenance can settle.

## Environment check at capture time

- Affinity was not running.
- `localhost:6767` was closed, so the MCP server was off (its default state).
- This agent (opencode) had no MCP servers configured.

## Traps

- **`.8bf` Photoshop plugins are a dead end.** Affinity's plugin support is
  limited to 8bf filters and does not follow Adobe's plugin SDK; vendors report
  their plugins do not run. Wrapping the model as an 8bf filter is not a route.
- **"Affinity MCP" is mostly the wrong product.** Most servers found under that
  name are for *Affinity CRM* (`affinity.co`), which has nothing to do with the
  design suite. Anything asking for an `sk-...` API key is the CRM one.

## Planned future work

**Sequence as of 2026-10-18** — to be taken in order, unless expedience says
otherwise. The contract in [`INTERFACE.md`](INTERFACE.md) is settled first, because
both programs have to satisfy it and the handler has to be written against it:

0. **The contract is written** — [`INTERFACE.md`](INTERFACE.md): invocation, result
   naming, the shared /8 frame, bit depth, the JSON manifest, failure and exit codes.
   Two small gaps remain on the deblur side (a `--version` flag and writing the
   manifest); both change no pixels and are worth closing before step 5.
1. **Enable an interface.** Turn on the MCP server in Affinity's Settings (3.2 or
   later; Affinity restarts), or take the more general JavaScript scripting route.
   The MCP server is the smaller step and is what makes the interface inspectable, so
   it goes first — see "Inspecting the interface" below.
2. **Make IFAN talk to the interface.** What this involves depends on the answer to
   "The open question" above:
   - **If export and place both exist:** no Affinity script is needed. The agent
     exports the active document, runs `local_infer.py` on it, and places the result
     back as a layer — landing straight in the mask-and-blend workflow.
   - **If they do not:** the manual path already works today (export from Affinity →
     run `run_infer.bat` or `local_infer.py` → place the `_db` file as a layer).
     Integration would then only save a few clicks.
3. **16-bit: done.** The deblur runner reads a file's own bit depth and writes the same
   one back with no arguments, per file, in a mixed folder — `LOCAL_SETUP.md` §5.6.
   `ckpt/IFAN_16bit.pytorch` is in place and the 16-bit path is verified, so there is
   nothing left to try here. This was the earlier step 3.
4. **Bring the depth map program up to the same contract.** A `local_depth.py` in the
   **local** tree of that project (not this repo, and not its GitHub copy), giving it
   what it lacks today: a command line, folder batches, results beside the originals
   with its own subfolder and suffix, the same shared /8 frame so its layer and the
   deblur layer register exactly, 16-bit greyscale by default, and the same manifest.
   Its working resolution comes from the model and must be **stated, not silent**. A
   tiled native-resolution pass is a later question — the model returns per-image
   relative depth, so tiles can disagree in scale and offset, and that needs measuring
   against seams rather than assuming.
5. **Then the handler, combining both.** One program, driving the two tools as
   processes and reading their manifests only — no imports, no parsing console text, no
   knowledge of which model ran. It imports each result as a layer at native size and
   leaves blending to the artist, as above. Because it shares no code with either tool,
   its own licence is free of theirs (see the licence section of `INTERFACE.md`).

Steps 1 and 2 can run in parallel with 4; step 5 needs 0 and 4 settled, and is much
easier once the manifest exists on both sides.

### Inspecting the interface (step 1)

- Dump the SDK docs locally (`node extract_docs.js`) and search them for *file*,
  *path*, *process*, *export* and *place*/*import*.
- Enumerate the live tool list with MCP Inspector, looking specifically for: export
  document, place/import image as layer, and any process execution.
- Register the server with opencode, proxying to stdio if SSE misbehaves.

### On the memory concern (step 3)

Bit depth does **not** change GPU memory. `in_bit` only affects how the input file is
read and what it is divided by (`norm_val`: 255 against 65535); the network runs in
`float32` either way. `config_IFAN.py` and `config_IFAN_16bit.py` are architecturally
identical — `N = 17`, `ch = 32`, `ks = 3`, `res_num = 2`, `refine_val = 8` — so the
15,232-channel tensor that sets the cost (`models/archs/IFAN.py`, `self.F`) is the
same size in both. The measured 1884 MiB peak and the ~1.8 GB per megapixel
whole-image figure apply unchanged. Image size is what costs memory, and tiling
already handles that.

Host RAM, not GPU memory, is what an image really costs: about 1.2 GiB of runtime
plus ~56 bytes per pixel, so the 20.9 Mpx frame peaks near 2.3 GiB
(`LOCAL_SETUP.md` §6). That is what `--max-pixels` guards, and its default follows
the free memory of whatever machine it is running on rather than a fixed count.

The three traps that shaped the 16-bit path are now handled in `local_infer.py`, and
are recorded here because they are what the code is defending against:

- **Config and file bit depth must match, in both directions.** The 16-bit branch is
  selected by `norm_val == 65535`, so an 8-bit file read by the 16-bit config would
  be divided by 65535 and come back almost black; the runner now refuses that with a
  message naming the fix. The reverse mismatch is the silent flattening already
  noted, and the runner warns about it once per run.
- **Alpha breaks the 16-bit read.** `cv2.imread(path, -1)` (`IMREAD_UNCHANGED`)
  returns the file as stored, so a 16-bit image carrying an alpha channel arrives as
  4 channels and `read_frame`'s `frame[...,::-1]` then reorders them wrongly. The
  16-bit path takes alpha off before the reversal instead, which is testable: the same
  image with and without alpha produces byte-identical output. The alpha is **not** put
  back on the result — a condition of the output, not a warning, since the original file
  still has its transparency and a correction layer wants the whole frame. A grey+alpha
  file is separated the same way, which upstream would get wrong by reversing its
  two-channel last axis.
- Model and config must be switched together — selecting `--config
  config_IFAN_16bit` switches both, so they cannot drift apart by accident.

## Related work — the 16-bit depth map project

A second project of mine targets the same audience: **`RRaz/DepthAnythingAC`**, a
fork of *Depth Anything At Any Condition*, worked on at
`H:\pythondepth\DepthAnything-AC`. It runs locally through a browser interface and
writes the depth map out as **16-bit greyscale**.

Checked 2026-10-18: the working tree and the GitHub fork are **in step** — `main` at
`3c0b280`, working tree clean, and this is the only copy of the project on the machine.
So there is no longer a lagging copy to worry about; the local tree is what runs, and it
is where a batch runner would be added.

Notes:

- **The output is a log, not a result.** `app.py` writes `logs/raw_depth_<timestamp>.png`
  as a side effect of showing a preview, so nothing connects a depth map to the file it
  came from. That is the first thing `INTERFACE.md` fixes.
- **The working resolution is the model's.** The short side is scaled to 518 px and the
  map is interpolated back to the original size, so the result matches the input's
  dimensions but is not at the input's detail. That has to be stated on the console
  rather than left to be discovered.
- **The environment is CPU-only**, which is why GPU is off: `.venv` holds
  `torch 2.3.0+cpu`, so `torch.cuda.is_available()` is `False` and the app takes the CPU
  path. At 518 px it is quick enough that this has never needed fixing. The
  `.venv-xformers` and `.venv-xformers-cu` folders are the abandoned GPU attempts —
  hard-referenced driver versions, rather than anything wrong with the model.
- A depth map is single-channel, so 16-bit greyscale is the most it can use — the
  48-bit question above belongs to the colour project, not this one.
- **Licence: CC-BY-NC-4.0 — non-commercial.** Checked rather than assumed: the upstream
  Space declares `cc-by-nc-4.0`, the vendored DINOv2 copy is CC-BY-NC-4.0 for code *and*
  weights, and the Depth Anything V2 code underneath is Apache-2.0. The fork now carries
  all of that in `LICENSE.md`, `THIRD_PARTY_NOTICES.md` and `LICENSES/`. This is the
  reason the Affinity handler is a contract rather than a shared library — see the
  licence section of `INTERFACE.md`.
- **Integrate against this tree**, and give it its own `local_depth.py` satisfying
  `INTERFACE.md` — planned step 4 above.

## Sources

- Scripting in Affinity — <https://www.affinity.studio/help/scripting-in-affinity/>
- AI Automation with Claude / AI connector setup —
  <https://www.affinity.studio/help/ai-connector-setup/>
- Scripts panel — <https://www.affinity.studio/help/panels-scripts-panel/>
- <https://github.com/rabidgremlin/affinity-scripting>
- <https://github.com/JiriKrblich/Affinity-script-manager>
