"""Local batch inference for IFAN (single image defocus deblurring).

A standalone replacement for predict.py that does not need the `cog`/Replicate
runtime and processes whole folders of images:

    python local_infer.py --input <folder>

Results land beside the input, in a `deblurred` subfolder and with a `_db` suffix,
so `photos\IMG_0001.jpg` becomes `photos\deblurred\IMG_0001_db.jpg`.  See --dest to
change that.

Why this exists
---------------
The upstream project shrinks any input whose longest side is over 1920 px before
deblurring (predict.py: `max_side = 1920`), so large images come back *smaller
than they went in*.  Raising that number alone is not enough: IFAN's filter
predictor emits a 15,232-channel tensor at 1/8 resolution, so peak memory grows
at roughly 1.8 GB per megapixel.  A 5760x3840 input needs a single 19.6 GiB
allocation and cannot fit on a 12 GB GPU at all.

So this runner never downscales.  Instead it:

  * splits anything larger than one tile into overlapping tiles, runs the
    network tile by tile, and feathers the results back together, so peak GPU
    memory depends on --tile rather than on the size of the image, and
  * refuses images above a pixel ceiling with a message that names the limit,
    rather than quietly shrinking them.

No arguments are needed at all.  The options listed by --help exist for tuning
only; every one of them has a working default.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

from configs.config_IFAN import get_config as get_config_8bit
from configs.config_IFAN_16bit import get_config as get_config_16bit
from data_loader.utils import read_frame, refine_image
from models.archs.IFAN import Network

DEFAULT_MAX_SIDE = 0               # 0 = never downscale (upstream capped at 1920)
DEFAULT_TILE = 1024                # largest tile side; 0 disables tiling
DEFAULT_OVERLAP = 256              # feather width between neighbouring tiles
DEFAULT_MAX_PIXELS = 100_000_000   # ceiling, refused with a message naming the limit
DEFAULT_DEST = 'both'              # 'output', 'subfolder', 'suffix' or 'both'
DEFAULT_SUBFOLDER = 'deblurred'    # folder created beside the inputs
DEFAULT_SUFFIX = '_db'             # appended to each output file name
DEFAULT_CONFIG = 'config_IFAN'     # 8-bit in and out, the published weights
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

# Configurations this runner knows how to pair with a checkpoint: name -> (mode for
# get_config, loader).  The two differ only in `in_bit` (8 against 16) and in the data
# their weights were trained on; the architecture is identical, so either checkpoint
# loads into either configuration.
CONFIGS = {
    'config_IFAN':       ('IFAN',       get_config_8bit),
    'config_IFAN_16bit': ('IFAN_16bit', get_config_16bit),
}
# Containers that can hold more than 8 bits per channel.  JPEG cannot, which is why
# 16-bit output cannot simply keep the input extension the way the default does.
DEEP_EXTS = ('.tif', '.tiff', '.png')
# OpenCV gives the TIFF Deflate tag no name; 8 is COMPRESSION_ADOBE_DEFLATE, the widely
# supported identifier.  Its deprecated twin, 32946, writes the same bytes but makes
# libtiff warn on every write.
TIFF_DEFLATE = 8


def find_images(root, recursive):
    """Return image paths under `root` (a file or a directory)."""
    if os.path.isfile(root):
        return [root]

    paths = []
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in sorted(filenames):
                if name.lower().endswith(IMAGE_EXTS) and name != 'Thumbs.db':
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
                paths.append(path)
    return paths


def is_within(path, folder):
    """True when `path` lies inside `folder` (case-insensitively, drive-aware)."""
    path = os.path.abspath(path)
    folder = os.path.abspath(folder)
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:      # different drives on Windows
        return False


def load_network(ckpt_path, config, device):
    """Build the IFAN network and load `ckpt_path` into it.

    Published checkpoints were saved from a DataParallel/DeblurNet-wrapped
    model, so their keys carry a leading `module.` and `Network.` prefix, and
    they also contain the reblurring network used only during training.
    """
    net = Network(config)

    try:
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        raw = torch.load(ckpt_path, map_location='cpu')
    except Exception:  # checkpoint holds pickled objects, not just tensors
        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    for key in ('state_dict', 'net'):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]

    if not isinstance(raw, dict):
        raise RuntimeError('Unsupported checkpoint format in {}'.format(ckpt_path))

    def strip(state_dict, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()}

    # Discover the wrapper prefix from the checkpoint itself, e.g.
    # 'module.Network.conv1_1.0.weight' yields the candidates 'module.' and
    # 'module.Network.'; the spelling that matches the most weights wins.
    sample = next(iter(raw))
    parts = sample.split('.')
    candidates = [('as-is', raw)]
    for i in range(1, len(parts)):
        prefix = '.'.join(parts[:i]) + '.'
        candidates.append(("with '{}' stripped".format(prefix), strip(raw, prefix)))

    model_keys = set(net.state_dict().keys())
    best_label, best_state, best_hits = None, None, -1
    for label, state_dict in candidates:
        hits = len(model_keys & set(state_dict.keys()))
        if hits > best_hits:
            best_label, best_state, best_hits = label, state_dict, hits

    print('checkpoint : {} ({} tensors)'.format(ckpt_path, len(raw)))
    print('key format : {} -> {}/{} parameters matched'.format(best_label, best_hits, len(model_keys)))

    if best_hits == 0:
        raise RuntimeError('No checkpoint weights matched the network; refusing to run.')

    weights = {k: v for k, v in best_state.items() if k in model_keys}
    try:
        net.load_state_dict(weights, strict=True)
        print('weights    : loaded, every parameter matched')
    except RuntimeError:
        result = net.load_state_dict(weights, strict=False)
        print('weights    : loaded with strict=False (missing={})'
              .format(len(result.missing_keys)))
        if result.missing_keys:
            print('WARNING: {} parameters were not found in the checkpoint; the '
                  'result may be wrong.'.format(len(result.missing_keys)))

    return net.to(device).eval()


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def _ramp(length):
    """Smooth ramp rising from ~0 to ~1; pairs sum to exactly 1 when mirrored."""
    t = np.arange(1, length + 1, dtype=np.float32) / (length + 1)
    return t * t * (3.0 - 2.0 * t)


def _blend_weights(length, overlap, first, last):
    """1-D feathering window for one tile along one axis."""
    weights = np.ones(length, dtype=np.float32)
    if overlap > 0:
        ramp = _ramp(overlap)
        weights[:overlap] = ramp
        weights[length - overlap:] = ramp[::-1]
        # Outer edges of the image have no neighbour to blend with.
        if first:
            weights[:overlap] = 1.0
        if last:
            weights[length - overlap:] = 1.0
    return weights


def _run_network(net, patch, device):
    """Run one RGB float32 patch through the network; returns (h, w, 3) float32."""
    tensor = torch.FloatTensor(patch.transpose(2, 0, 1)[np.newaxis].copy()).to(device)
    with torch.no_grad():
        result = net(tensor)['result']
    return result[0].permute(1, 2, 0).cpu().numpy()


def _tile_starts(length, tile, stride):
    """Tile origins covering `length`, ending flush with the far edge."""
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def deblur_image(net, config, device, image, tile, overlap):
    """Deblur an (h, w, 3) float32 RGB image, tiling when it exceeds `tile`.

    Returns the blended result and the number of tiles used.
    """
    height, width = image.shape[0], image.shape[1]
    step = config.refine_val  # network downsamples by this factor

    if tile <= 0 or (height <= tile and width <= tile):
        return _run_network(net, image, device), 1

    # Keep tile geometry aligned to the network's stride so that features line
    # up between neighbouring tiles, and never let the feather overlap half a tile.
    tile -= tile % step
    overlap = min(overlap - overlap % step, tile // 2)
    stride = tile - overlap

    ys = _tile_starts(height, tile, stride)
    xs = _tile_starts(width, tile, stride)

    accum = np.zeros((height, width, 3), dtype=np.float32)
    weight_sum = np.zeros((height, width, 1), dtype=np.float32)

    for row, y in enumerate(ys):
        tile_h = min(tile, height - y)
        weight_h = _blend_weights(tile_h, overlap, row == 0, row == len(ys) - 1)
        for col, x in enumerate(xs):
            tile_w = min(tile, width - x)
            weight_w = _blend_weights(tile_w, overlap, col == 0, col == len(xs) - 1)
            weight = np.outer(weight_h, weight_w)[:, :, None]
            result = _run_network(net, image[y:y + tile_h, x:x + tile_w], device)
            accum[y:y + tile_h, x:x + tile_w] += result * weight
            weight_sum[y:y + tile_h, x:x + tile_w] += weight

    return accum / np.maximum(weight_sum, 1e-6), len(ys) * len(xs)


def input_bit_depth(path):
    """Bits per channel in `path`, or None if it cannot be read.

    Formats that cannot hold more than 8 bits are answered without opening the file;
    the rest are probed.  This only ever feeds a warning -- it never changes how a
    file is read.
    """
    if os.path.splitext(path)[1].lower() not in DEEP_EXTS:
        return 8
    probe = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if probe is None:
        return None
    return probe.dtype.itemsize * 8


def read_input(path, config):
    """Read `path` as (1, h, w, 3) float RGB in [0, 1].

    The 8-bit path is upstream's `read_frame`, untouched.  The 16-bit path reads at
    native depth, which is what upstream's 16-bit branch does too, except that it
    drops an alpha channel first: upstream reverses the last axis, which would turn a
    four-channel BGRA image into ARGB and quietly corrupt it, and Affinity TIFF
    exports commonly carry alpha.

    A DNG is read as a bare TIFF, since that is all OpenCV sees.  It says so rather
    than pretending: everything that makes a DNG a DNG is skipped.
    """
    if os.path.splitext(path)[1].lower() == '.dng':
        print('WARNING    : {} is a DNG, read as a bare TIFF: its colour matrix, white '
              'balance and linearisation are skipped, so it will not match the same '
              'picture exported as TIFF or PNG'.format(os.path.basename(path)))

    if getattr(config, 'in_bit', 8) != 16:
        return read_frame(path, config.norm_val, None)

    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError('could not read {}'.format(path))
    if raw.dtype != np.uint16:
        raise RuntimeError(
            '{} holds {}-bit data but this config reads 16-bit, and dividing it by '
            '65535 would bring it back almost black; use --config config_IFAN for '
            '8-bit files'.format(path, raw.dtype.itemsize * 8))

    if raw.ndim == 3 and raw.shape[2] == 4:
        raw = raw[:, :, :3]                        # drop alpha before reversing
    if raw.ndim == 2:
        print('WARNING    : {} is single-channel, so it is treated as greyscale; if this '
              'is a raw file it is a colour-filter array (a mosaic) and the result is '
              'meaningless'.format(os.path.basename(path)))
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    else:
        raw = raw[:, :, ::-1]                      # BGR -> RGB

    return np.expand_dims(raw / 65535.0, axis=0)


def to_output(blended, in_bit):
    """(h, w, 3) float RGB in [0, 1] -> BGR array at the config's bit depth."""
    scale = (2 ** in_bit) - 1
    dtype = np.uint16 if in_bit == 16 else np.uint8
    return (np.flip(blended, 2) * scale).astype(dtype)   # RGB -> BGR for OpenCV


def process(net, config, device, path, output_path, max_side, tile, overlap, max_pixels):
    """Deblur a single image and write it to `output_path`."""
    frame = read_input(path, config)   # (1, h, w, 3) float RGB in [0, 1]
    _, height, width, _ = frame.shape

    note = ''
    if max_side and max(height, width) > max_side:
        ratio = max_side / max(height, width)
        scaled = (int(width * ratio), int(height * ratio))
        frame = np.expand_dims(
            cv2.resize(frame[0], dsize=scaled, interpolation=cv2.INTER_AREA), 0)
        note = ' downscaled {}x{}->{}x{}'.format(width, height, scaled[0], scaled[1])

    # Past the ceiling we stop rather than quietly produce a smaller picture.
    pixels = frame.shape[1] * frame.shape[2]
    if max_pixels and pixels > max_pixels:
        raise RuntimeError(
            '{}x{} = {:.1f} megapixels is over the {:.1f} megapixel limit '
            '(raise it with --max-pixels if you have the RAM and time)'.format(
                frame.shape[2], frame.shape[1], pixels / 1e6, max_pixels / 1e6))

    # The network downsamples by 8, so trim to a multiple of config.refine_val.
    image = refine_image(frame, config.refine_val)[0]
    out_h, out_w = image.shape[0], image.shape[1]
    if out_h < config.refine_val or out_w < config.refine_val:
        raise RuntimeError('image too small after trimming ({}x{})'.format(out_w, out_h))

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    blended, tiles = deblur_image(net, config, device, image, tile, overlap)
    elapsed = time.time() - started

    output = to_output(blended, getattr(config, 'in_bit', 8))   # RGB -> BGR

    # OpenCV defaults TIFF to LZW, which *inflates* 16-bit output: 147 MiB for data
    # that is 119 MiB raw, because the low byte of a smooth 16-bit gradient is close
    # to random.  Adobe Deflate is lossless and gets the same picture into 108 MiB.
    compressed = (getattr(config, 'in_bit', 8) == 16
                  and output_path.lower().endswith(('.tif', '.tiff')))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if compressed:
        written = cv2.imwrite(output_path, output,
                              [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_DEFLATE])
    else:
        written = cv2.imwrite(output_path, output)
    if not written:
        raise RuntimeError('could not write {}'.format(output_path))

    return elapsed, (width, height), (out_w, out_h), note, tiles


def main():
    parser = argparse.ArgumentParser(
        description='IFAN defocus deblurring: local batch runner for folders of images.')
    parser.add_argument('--input', '-i', required=True, help='input image or folder')
    parser.add_argument('--dest', choices=['output', 'subfolder', 'suffix', 'both'], default=None,
                        help='where results go: "output" writes into --output; "subfolder" '
                             'writes into <input>/<--subfolder>; "suffix" writes beside each '
                             'input as <name><--suffix><ext>; "both" does the last two '
                             '(default: %s)' % DEFAULT_DEST)
    parser.add_argument('--subfolder', default=DEFAULT_SUBFOLDER,
                        help='subfolder name used by --dest subfolder and both '
                             '(default: %(default)s)')
    parser.add_argument('--suffix', default=DEFAULT_SUFFIX,
                        help='file-name suffix used by --dest suffix and both '
                             '(default: %(default)s)')
    parser.add_argument('--output', '-o', default=None,
                        help='destination root, used by --dest output '
                             '(default: <project>\\output)')
    parser.add_argument('--ckpt', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'ckpt', 'IFAN.pytorch'),
                        help='path to the pretrained checkpoint')
    parser.add_argument('--max-side', type=int, default=DEFAULT_MAX_SIDE,
                        help='downscale inputs whose longest side exceeds this many pixels; '
                             '0 (the default) never downscales')
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS,
                        help='refuse images above this many pixels instead of shrinking them; '
                             '0 removes the ceiling (default: %(default)s)')
    parser.add_argument('--tile', type=int, default=DEFAULT_TILE,
                        help='largest tile side when processing large images; '
                             '0 disables tiling and needs memory proportional to the image '
                             '(default: %(default)s)')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP,
                        help='feather width between tiles (default: %(default)s)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='recurse into subfolders, keeping the folder structure')
    parser.add_argument('--cpu', action='store_true', help='run on CPU even if a GPU is available')
    parser.add_argument('--config', default=DEFAULT_CONFIG, choices=list(CONFIGS),
                        help='model configuration: config_IFAN (default) reads and writes '
                             '8-bit; config_IFAN_16bit reads 16-bit input and writes 16-bit '
                             'output, which needs --ext tif or png')
    parser.add_argument('--ext', default='keep', choices=['keep', 'png', 'jpg', 'tif'],
                        help='output format; "keep" preserves each input extension (default)')
    args = parser.parse_args()

    # Naming --output is shorthand for --dest output.
    if args.dest is None:
        args.dest = 'output' if args.output is not None else DEFAULT_DEST
    if args.dest == 'output' and args.output is None:
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

    if not os.path.exists(args.input):
        sys.exit('Input not found: {}'.format(args.input))
    if not os.path.isfile(args.ckpt):
        sys.exit('Checkpoint not found: {}\nDownload it from the README links and place it there.'
                 .format(args.ckpt))

    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
    mode, load_config = CONFIGS[args.config]
    config = load_config('IFAN_CVPR2021', mode, args.config)
    config.network = 'IFAN'
    config.device = str(device)

    tile = args.tile if args.tile <= 0 else args.tile - args.tile % config.refine_val
    if args.tile > 0 and tile != args.tile:
        print('note       : tile rounded down to {} (multiple of {})'
              .format(tile, config.refine_val))
    overlap = args.overlap - args.overlap % config.refine_val

    print('device     : {}{}'.format(device, '' if device.type == 'cpu' else
                                     ' ({})'.format(torch.cuda.get_device_name(0))))
    print('model      : {} ({}-bit in and out)'.format(args.config, config.in_bit))
    print('max side   : {}'.format('unlimited' if not args.max_side else args.max_side))
    print('tiling     : {}'.format('disabled' if tile <= 0 else
                                   '{} px tiles, {} px overlap'.format(tile, overlap)))
    print('pixel limit: {}'.format('none' if not args.max_pixels else
                                   '{:.0f} megapixels'.format(args.max_pixels / 1e6)))

    net = load_network(args.ckpt, config, device)

    # Results are written relative to the input: into a subfolder of it, beside each
    # image with a suffix, or both.  "output" sends them to a separate root instead.
    root = args.input if os.path.isdir(args.input) else os.path.dirname(os.path.abspath(args.input))
    subfolder = os.path.join(root, args.subfolder)
    in_subfolder = args.dest in ('subfolder', 'both')
    beside = args.dest in ('suffix', 'both')

    images = find_images(args.input, args.recursive)
    # Never pick up our own earlier results, or a second run would deblur the
    # deblurred copies and the folder would grow on every run.
    skipped = 0
    if in_subfolder or beside:
        kept = []
        for path in images:
            if in_subfolder and is_within(path, subfolder):
                skipped += 1
            elif (beside and args.suffix
                  and os.path.splitext(os.path.basename(path))[0].endswith(args.suffix)):
                skipped += 1
            else:
                kept.append(path)
        images = kept
    if not images:
        if skipped:
            print('Nothing to do: all {} image(s) under {} are already processed'
                  .format(skipped, args.input))
            return
        sys.exit('No images found under {}'.format(args.input))

    # A DNG can be read here but not written, so it can never keep its own extension.
    if args.ext == 'keep':
        raws = [p for p in images if os.path.splitext(p)[1].lower() == '.dng']
        if raws:
            sys.exit('DNG cannot be written back out, only read; add --ext tif (or png) '
                     'to choose a container for the result ({}).'
                     .format(os.path.basename(raws[0])))

    # 16-bit results need a container that can hold them; JPEG and WebP cannot.
    if getattr(config, 'in_bit', 8) > 8:
        if args.ext != 'keep' and '.' + args.ext not in DEEP_EXTS:
            sys.exit('--ext {} cannot hold {}-bit output.  Use --ext tif (or png), or run '
                     'with --config config_IFAN for 8-bit results.'
                     .format(args.ext, config.in_bit))
        if args.ext == 'keep':
            blocked = [p for p in images
                       if os.path.splitext(p)[1].lower() not in DEEP_EXTS]
            if blocked:
                sys.exit('{} of {} input(s) end in a format that cannot hold {}-bit output '
                         '({}).  Use --ext tif, or run with --config config_IFAN.'
                         .format(len(blocked), len(images), config.in_bit,
                                 os.path.basename(blocked[0])))
    else:
        probe = next((p for p in images
                      if os.path.splitext(p)[1].lower() in DEEP_EXTS), None)
        if probe and (input_bit_depth(probe) or 0) > 8:
            print('WARNING    : {} holds 16-bit data but this config reads 8-bit, so the '
                  'extra precision is discarded; pass --config config_IFAN_16bit to keep '
                  'it'.format(os.path.basename(probe)))

    if args.dest == 'output':
        destination = os.path.abspath(args.output)
    elif args.dest == 'subfolder':
        destination = subfolder
    elif args.dest == 'suffix':
        destination = 'beside each input, as <name>{}<ext>'.format(args.suffix)
    else:
        destination = '{} as <name>{}<ext>'.format(subfolder, args.suffix)

    print('images     : {} found under {}'.format(len(images), args.input))
    if skipped:
        print('skipped    : {} already-processed file(s)'.format(skipped))
    print('destination: {}'.format(destination))
    print('')

    total = 0.0
    done = 0
    failed = 0
    for index, path in enumerate(images, 1):
        relative = os.path.relpath(path, root)
        directory, filename = os.path.split(relative)
        stem, input_ext = os.path.splitext(filename)
        ext = input_ext if args.ext == 'keep' else '.' + args.ext
        name = stem + (args.suffix if beside else '') + ext
        if args.dest == 'suffix':
            output_path = os.path.join(root, directory, name)
        else:
            output_path = os.path.join(subfolder if in_subfolder else args.output,
                                       directory, name)

        try:
            elapsed, original, processed, note, tiles = process(
                net, config, device, path, output_path,
                args.max_side, tile, overlap, args.max_pixels)
        except Exception as ex:
            failed += 1
            print('[{:>3}/{}] FAILED {}: {}'.format(index, len(images), relative, ex))
            continue

        done += 1
        total += elapsed
        if device.type == 'cuda':
            peak = ' peak GPU {:.0f} MiB'.format(torch.cuda.max_memory_allocated() / 1024 ** 2)
        else:
            peak = ''
        tiled = ' [{} tiles]'.format(tiles) if tiles > 1 else ''
        print('[{:>3}/{}] {}  {}x{} -> {}x{}{}{}  {:.2f}s{}'
              .format(index, len(images), relative, original[0], original[1],
                      processed[0], processed[1], note, tiled, elapsed, peak))

    print('')
    summary = 'Done: {} of {} image(s) in {:.1f}s'.format(done, len(images), total)
    if done:
        summary += ' (average {:.2f}s)'.format(total / done)
    if failed:
        summary += ', {} failed'.format(failed)
    print(summary)


if __name__ == '__main__':
    main()
