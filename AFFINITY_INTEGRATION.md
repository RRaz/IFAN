# Driving IFAN from Affinity 3 — open question

**Status:** not started · captured 2026-10-18

An idea to pick up later: can Affinity 3 (by Canva) hand an image to this project
and take the deblurred result back — as a *layer*, not a flattened file?

## Why it is worth doing

Photographers already working in Affinity 3 would get deblurring as a step inside
their normal edit. The workflow this repo is built for is mask-and-blend **by eye**:
the deblurred plate is composited back over the original and a mask decides where
it applies, so sky and distant objects keep the softer rendering they should have.
Uniform deblurring across the frame fights how a photograph is read.

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

## Next steps

1. Enable the MCP server in Affinity's Settings (3.2 or later) and let it restart.
2. Dump the SDK docs locally (`node extract_docs.js`) and search them for *file*,
   *path*, *process*, *export* and *place*/*import*.
3. Enumerate the live tool list with MCP Inspector, looking specifically for:
   export document, place/import image as layer, and any process execution.
4. Register the server with opencode, proxying to stdio if SSE misbehaves.
5. **If export and place both exist:** no Affinity script is needed. The agent
   exports the active document, runs `local_infer.py` on it, and places the result
   back as a layer — landing straight in the mask-and-blend workflow.
6. **If they do not:** the manual path already works today (export from Affinity →
   run `run_infer.bat` or `local_infer.py` → place the `_db` file as a layer).
   Integration would then only save a few clicks.

## Related work

The same integration would suit a separate **16-bit depth map modification**, to be
written up when picked up — also aimed at Affinity 3 users.

Relevant to that: upstream ships a **16-bit trained variant**, so a 16-bit
workflow need not round-trip through 8-bit:

```
python run.py --mode IFAN_16bit --network IFAN --config config_IFAN_16bit \
  --data DPDD --ckpt_abs_name ckpt/IFAN_16bit.pytorch \
  --data_offset /data_offset --output_offset ./output
```

(README, "Our model trained with 16 bit images" — Table 1 of the supplementary
material.) Its weights are separate from `ckpt/IFAN.pytorch`.

## Sources

- Scripting in Affinity — <https://www.affinity.studio/help/scripting-in-affinity/>
- AI Automation with Claude / AI connector setup —
  <https://www.affinity.studio/help/ai-connector-setup/>
- Scripts panel — <https://www.affinity.studio/help/panels-scripts-panel/>
- <https://github.com/rabidgremlin/affinity-scripting>
- <https://github.com/JiriKrblich/Affinity-script-manager>
