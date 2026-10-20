# Driving IFAN from Affinity 3 — open question

**Status:** add-on, step 5 only · captured 2026-10-18, updated 2026-10-20

**The interface it will be written against is fixed, and both tools satisfy it:**
[`INTERFACE.md`](INTERFACE.md). `local_infer.py` (deblur) and `local_depth.py` (depth,
in that project's tree) both answer `--version`, write a contract-1 manifest and fail per
file without stopping a batch. What is left is the adapter that Affinity calls — step 5
below — and the pixel filter handshake it has to satisfy, under "The pixel filter route".

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

**Three interfaces exist.**

0. **Pixel filter** — an outside program that receives the current pixel layer and
   returns the replacement for it. Older than the two below, and the one that fits this
   job; see "The pixel filter route".

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

## The pixel filter route

**Reported 2026-10-20.** Affinity can call an outside program as a **pixel filter**: it
hands the current pixel layer to that program and takes the image it returns as the
replacement content of the same layer. It is an older mechanism than JavaScript
scripting, and it answers both halves of the original question in one step — so the
export-then-place route, and the uncertainty about whether scripting may touch arbitrary
paths, both drop out.

If it works as described, the adapter is not a document driver but a **thin program**:

1. Affinity hands it the layer as an image file.
2. It runs `local_infer.py` (or `local_depth.py`) on that file with `--manifest` and
   reads the `output` path from the manifest.
3. It returns that image; Affinity replaces the layer with it. The layer is now
   deblurred, or has become a depth map, and masking and opacity stay with the artist.

What goes away: exporting the document, placing a layer, file pickers, the MCP server,
and the question of whether a script may spawn a process. What replaces it: the returned
image has to be exactly the layer's size, because the layer is being *replaced* rather
than fitted into a document.

Three things decide the adapter's shape, and nothing else does:

- **What is passed.** A temporary file? A path in the arguments? Which format, colour
  profile and alpha? Does the layer arrive flattened, or with its own transparency?
- **Bit depth.** Does the handshake carry 16 bits per channel? If it is 8-bit, this
  route loses the precision that makes a partial-opacity blend work (`LOCAL_SETUP.md`
  §5.6), and the depth map arrives quantised to 8 bits. That is the one finding that
  would send the work back to the file-based route.
- **Size.** Must the returned image match the layer exactly? The deblur network crops to
  a multiple of 8, so its result can be up to 7 px smaller (`INTERFACE.md` §3). The
  adapter would then extend the missing bottom/right strip by repeating the last row and
  column — no resampling, no invented detail — rather than let Affinity scale the layer.
  A depth map needs nothing: it is exactly the input's size.

A **probe** settles all three in one run: a program registered as the filter that writes
down its argument list, working directory, and everything it can learn about the file it
was handed — dimensions, bit depth, channels, alpha, profile — then returns that same
file unchanged, so the layer comes back as it went in.

Do **not** assume an API such as `api.runProcess` carries over from other applications —
that call belongs to Cavalry, not Affinity.

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

**Sequence as of 2026-10-20** — to be taken in order, unless expedience says otherwise.
The contract in [`INTERFACE.md`](INTERFACE.md) was settled first, because both programs
have to satisfy it and the handler has to be written against it:

0. **The contract is written, and both sides satisfy it** —
   [`INTERFACE.md`](INTERFACE.md): invocation, result naming, geometry, bit depth, the
   JSON manifest, failure and exit codes. The two gaps that stood here (`--version` and
   the manifest) were closed on the deblur side on 2026-10-19 and on the depth side on
   2026-10-20.
1. **Establish the pixel filter handshake.** Register the probe program as a filter, run
   it on a known 16-bit layer, and read what it recorded. That settles format, bit depth,
   size and how parameters reach the program — see "The pixel filter route". The MCP
   server and the JavaScript API remain available and are worth inspecting for other
   jobs, but the filter route does not depend on them.
2. **Write the adapter.** One program with two entry points — deblur and depth: take the
   file Affinity passed, run the tool as a process with `--manifest`, return `output` at
   the layer's dimensions. If the handshake turns out to be 8-bit only, the file-based
   route stays the way to get 16-bit results, and the manual path (export → run
   `run_infer.bat` / `run_depth.bat` → place the `_db` or `_depth` file) remains the
   fallback.
3. **16-bit: done.** The deblur runner reads a file's own bit depth and writes the same
   one back with no arguments, per file, in a mixed folder — `LOCAL_SETUP.md` §5.6.
   `ckpt/IFAN_16bit.pytorch` is in place and the 16-bit path is verified, so there is
   nothing left to try here. This was the earlier step 3.
4. **The depth tool now satisfies it — done 2026-10-20.** `local_depth.py` in the depth
   tree (`H:\pythondepth\DepthAnything-AC`, pushed to `RRaz/DepthAnythingAC`) gives it
   the command line, folder batches, `depth/` plus `_depth`, 16-bit greyscale TIFF,
   per-file failures, `--version` and the same manifest. Two differences from the plan
   as first written: the map is **exactly the input's size** rather than /8 cropped —
   its network resamples internally and returns the map to the full frame, which is what
   a mask for the document wants, and `INTERFACE.md` §3 now states it — and an 8-bit
   colour preview (PNG, `Spectral_r`) is written beside each map and named in the
   manifest's optional `color_map`, for the eye only. The model's 518 px working size is
   printed on every file line rather than left to be discovered.
5. **Then the adapter, serving both tools.** One program, driving the two tools as
   processes and reading their manifests only — no imports, no parsing console text, no
   knowledge of which model ran. It is registered with Affinity as a pixel filter, so it
   returns an image and Affinity puts it back in the layer; blending stays with the
   artist, as above. Because it shares no code with either tool, its own licence is free
   of theirs (see the licence section of `INTERFACE.md`).

Steps 1 and 2 are the whole of the Affinity work and can be done whenever Affinity is
open in front of you; step 5 needs 1 and 4 settled.

### Inspecting the interface (step 1)

- Find the pixel filter command in Affinity 3 and register the probe as a filter; the
  probe records what it was handed, so the handshake is discovered rather than assumed.
- If the filter route turns out to have a gap, fall back to inspecting the JavaScript
  side: dump the SDK docs locally (`node extract_docs.js`) and search them for *file*,
  *path*, *process*, *export* and *place*/*import*; enumerate the live MCP tool list with
  MCP Inspector, looking for export, place-as-layer and process execution.
- Register the MCP server with opencode, proxying to stdio if SSE misbehaves.

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
`H:\pythondepth\DepthAnything-AC`. It runs locally through a browser interface, and
since 2026-10-20 through `local_depth.py` as well, and writes the depth map out as
**16-bit greyscale**.

Checked 2026-10-18: the working tree and the GitHub fork are **in step** — this is the
only copy of the project on the machine. The local tree is what runs, and it is where
the batch runner was added (step 4 above, done).

Notes:

- **The output is now a result, not a log.** `local_depth.py` writes
  `<name>_depth.tif` beside the source in `depth/`, with `_depth` in the name and a path
  in the manifest, so a map is always traceable to the file it came from. The browser
  app still writes `logs/raw_depth_<timestamp>.png` as a side effect of showing a
  preview, exactly as before.
- **The working resolution is the model's.** The short side is scaled to 518 px and the
  map is interpolated back to the original size, so the result matches the input's
  dimensions exactly but is not at the input's detail. The runner states the working
  frame on every line rather than leaving it to be discovered.
- **The project is CPU only**, by its own declaration: there is no CUDA path left in
  `app.py` or in the runner, no GPU wheel in `requirements.txt`, and the two abandoned
  GPU environments (`.venv-xformers`, `.venv-xformers-cu`) have been deleted. `.venv`
  holds `torch 2.3.0+cpu`; a 16.9 Mpx frame takes about 0.74 s at 518 px.
- A depth map is single-channel, so 16-bit greyscale is the most it can use — the
  48-bit question above belongs to the colour project, not this one.
- **Licence: CC-BY-NC-4.0 — non-commercial.** Checked rather than assumed: the upstream
  Space declares `cc-by-nc-4.0`, the vendored DINOv2 copy is CC-BY-NC-4.0 for code *and*
  weights, and the Depth Anything V2 code underneath is Apache-2.0. The fork now carries
  all of that in `LICENSE.md`, `THIRD_PARTY_NOTICES.md` and `LICENSES/`. This is the
  reason the Affinity handler is a contract rather than a shared library — see the
  licence section of `INTERFACE.md`.
- **Integrate against this tree.** `local_depth.py` already satisfies `INTERFACE.md`, so
  the handler calls it here rather than reimplementing anything.

## Sources

- Scripting in Affinity — <https://www.affinity.studio/help/scripting-in-affinity/>
- AI Automation with Claude / AI connector setup —
  <https://www.affinity.studio/help/ai-connector-setup/>
- Scripts panel — <https://www.affinity.studio/help/panels-scripts-panel/>
- <https://github.com/rabidgremlin/affinity-scripting>
- <https://github.com/JiriKrblich/Affinity-script-manager>
