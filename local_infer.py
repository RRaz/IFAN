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
  * refuses images above a pixel ceiling that follows the machine's own memory,
    with a message that names the limit, rather than quietly shrinking them.

Bit depth follows the file: a 16-bit TIFF is read, deblurred and written at 16 bits
with no arguments, an 8-bit JPEG stays 8-bit, and a folder holding both gets the right
treatment for each.  --config forces one mode for a whole run when that is wanted.

An alpha channel is not carried through, by design: the original file still has it, and
what comes back is the whole frame.

A JPEG result keeps the input's own chroma sampling rather than OpenCV's default.  That
default is 4:2:0, and upstream's predict.py writes its results that way: on an ordinary
camera JPEG little is lost, but on a near-lossless 4:4:4 export -- an Affinity or
Lightroom master, say -- it discards colour detail the file already had, and that
difference measures about as large as half the deblurring itself.  So a 4:2:0 file comes
back 4:2:0 and a 4:4:4 file comes back 4:4:4, and a lossless input, having no sampling to
keep, is given 4:4:4 rather than being degraded on the way out.  16-bit TIFF output, which
is what to use when the result goes back into an editor, is unaffected: TIFF is lossless
either way.

`--manifest <path>` writes a JSON record of the run, and `--version` prints the tool id and
version that record carries.  INTERFACE.md is the contract for both; it is what lets a
second program drive this one without importing it.

No arguments are needed at all.  The options listed by --help exist for tuning
only; every one of them has a working default.
"""

import argparse
import ctypes
import json
import os
import sys
import time
from datetime import datetime

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
DEFAULT_MAX_PIXELS = 100_000_000   # only for a machine that will not say how much memory
                                   # it has; see `pixel_ceiling`
DEFAULT_DEST = 'both'              # 'output', 'subfolder', 'suffix' or 'both'
DEFAULT_SUBFOLDER = 'deblurred'    # folder created beside the inputs
DEFAULT_SUFFIX = '_db'             # appended to each output file name
DEFAULT_CONFIG = 'auto'            # match each file's own bit depth
# The interface contract this runner satisfies: the number a second program checks before
# it trusts anything else in a run record, and the tool id and version it looks for.
CONTRACT = 1
TOOL_ID = 'ifan-deblur'
TOOL_VERSION = '1.0.0'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
HERE = os.path.dirname(os.path.abspath(__file__))
# The network downsamples by this factor, so tile geometry and the output trim are
# aligned to it.  Both configurations set `refine_val` to this; it is mirrored here only
# so that tiles can be settled before a configuration is chosen.
REFINE_VAL = 8
# Each configuration has its own weights.  `IFAN_16bit.pytorch` is an optional extra
# download; the two architectures are identical, so the 8-bit checkpoint loads into the
# 16-bit configuration when the matching one is absent -- see `checkpoint_for`.
CKPTS = {
    'config_IFAN':       os.path.join(HERE, 'ckpt', 'IFAN.pytorch'),
    'config_IFAN_16bit': os.path.join(HERE, 'ckpt', 'IFAN_16bit.pytorch'),
}
FALLBACK_CKPT = os.path.join(HERE, 'ckpt', 'IFAN.pytorch')

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
# Extensions that are always 8 bits per channel, so their depth need never be probed.
# Anything else gets opened and asked -- a DNG included, which can be deep but is absent
# from IMAGE_EXTS because it cannot be written back out.
SHALLOW_EXTS = ('.jpg', '.jpeg', '.bmp', '.webp')
# OpenCV gives the TIFF Deflate tag no name; 8 is COMPRESSION_ADOBE_DEFLATE, the widely
# supported identifier.  Its deprecated twin, 32946, writes the same bytes but makes
# libtiff warn on every write.
TIFF_DEFLATE = 8
# OpenCV's JPEG writer defaults to 4:2:0, halving colour resolution, and it only grew the
# parameter to say otherwise in 4.5.5.  Its own constants are named where they exist and
# written out where they do not, so an older OpenCV falls back to the old behaviour
# instead of being handed a parameter it does not know.
JPEG_SAMPLING = getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR', None)
JPEG_SAMPLING_FACTORS = {
    '444': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_444', 0x111111),
    '422': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_422', 0x211111),
    '420': getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_420', 0x221111),
}


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


# What one deblur costs the machine, measured rather than guessed (LOCAL_SETUP.md §6):
# fitted over five sizes from 1 to 21 megapixels, it is linear to within half a percent at
# BYTES_PER_PIXEL per pixel plus RUNTIME_RAM.  It is the same at 8 and 16 bits.  Small as
# those bytes look, the whole image is resident several times over at the worst moment --
# upstream's read hands the frame back as float64 and the tile accumulator is float32 --
# which is why the ceiling below is a memory guard rather than an arbitrary count.
BYTES_PER_PIXEL = 56
RUNTIME_RAM = 1.2 * 1024 ** 3      # interpreter, torch, OpenCV and the CUDA context
MEMORY_SHARE = 0.5                 # of what is free; the rest of the machine keeps working


def memory_reading():
    """`(free, total)` physical memory in bytes, or None if the machine will not say.

    psutil answers if it happens to be installed, otherwise the platform is asked
    directly, so that running this needs nothing added to the environment.
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return int(memory.available), int(memory.total)
    except Exception:       # not installed, or it could not work out the answer
        pass

    if sys.platform == 'win32':
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys), int(status.ullTotalPhys)
        return None

    try:
        page = os.sysconf('SC_PAGE_SIZE')
        return page * os.sysconf('SC_AVPHYS_PAGES'), page * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, ValueError, OSError):
        return None


def image_ram(pixels):
    """Bytes of host RAM a `pixels` image takes, by the arithmetic the ceiling uses."""
    return RUNTIME_RAM + pixels * BYTES_PER_PIXEL


def pixel_ceiling():
    """The largest image this machine should be asked to hold, in pixels, or None.

    The ceiling guards host RAM, so it belongs to the machine rather than to this file:
    what a 32 GB workstation holds will not fit on a laptop.  None means the machine would
    not say, leaving the caller to fall back on the constant above.
    """
    reading = memory_reading()
    if not reading:
        return None
    free, _total = reading
    return max(1_000_000, int((free * MEMORY_SHARE - RUNTIME_RAM) / BYTES_PER_PIXEL))


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
    """Bits per channel stored in `path`, or None if it cannot be read.

    Formats that cannot hold more than 8 bits are answered from the extension without
    opening the file; everything else is probed.  `read_input` can also answer this while
    reading for real, and prefers to.
    """
    if os.path.splitext(path)[1].lower() in SHALLOW_EXTS:
        return 8
    probe = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if probe is None:
        return None
    return probe.dtype.itemsize * 8


def config_name_for_bit_depth(bits):
    """The configuration that reads and writes `bits` bits per channel."""
    return 'config_IFAN_16bit' if bits > 8 else 'config_IFAN'


def output_extension(input_ext, requested):
    """The extension to write a result with, and whether it had to be changed.

    `requested` is whatever --ext said.  "keep" means the input's own extension, except
    for a format this runner can read but not write back -- a DNG -- where TIFF stands
    in, because otherwise the result has nowhere to go and the run stops for a reason the
    user could do nothing about.
    """
    if requested != 'keep':
        return '.' + requested, False
    if input_ext.lower() in IMAGE_EXTS:
        return input_ext, False
    return '.tif', True


def container_error(bits, output_ext):
    """Why `output_ext` cannot hold `bits` bits per channel, or None if it can."""
    if bits > 8 and output_ext not in DEEP_EXTS:
        return ('{} cannot hold {}-bit output; use --ext tif (or png), or --config '
                'config_IFAN if 8-bit is what you want'
                .format(output_ext or 'that format', bits))
    return None


def checkpoint_for(config_name, requested):
    """Checkpoint to load for `config_name`, and a note when it is not the matching one.

    Each configuration has its own weights, but the two architectures are identical, so
    either checkpoint loads into either.  When the matching one is absent this falls back
    to the 8-bit checkpoint rather than refusing to run -- input and output still keep
    full precision, only the model's own correction is the 8-bit one -- and says so,
    because otherwise the run looks like it used weights that are not there.
    """
    if requested:
        return requested, None
    own = CKPTS[config_name]
    if os.path.isfile(own):
        return own, None
    if not os.path.isfile(FALLBACK_CKPT):
        raise RuntimeError(
            'no checkpoint found: neither {} nor {}.  Download them from the README '
            'links and place them in ckpt\\.'.format(own, FALLBACK_CKPT))
    return FALLBACK_CKPT, (
        '{} is not there, so the 8-bit checkpoint is used for {}: the correction is the '
        '8-bit model\'s, carried through at full precision.  Copy {} into ckpt\\ to use '
        'the weights trained for 16-bit.'
        .format(os.path.basename(own), config_name, os.path.basename(own)))


class Runner:
    """Loads one network per configuration, on demand, and reuses it.

    A folder can hold 8-bit and 16-bit files side by side and each wants its own
    configuration, so both networks may end up resident.  Their parameters are small
    next to the activations, which are per tile.
    """

    def __init__(self, device, requested_ckpt):
        self.device = device
        self.requested_ckpt = requested_ckpt
        self._loaded = {}
        self._noted = set()

    def get(self, config_name):
        """Return `(config, network)` for `config_name`, loading it the first time."""
        if config_name not in self._loaded:
            mode, load_config = CONFIGS[config_name]
            config = load_config('IFAN_CVPR2021', mode, config_name)
            config.network = 'IFAN'
            config.device = str(self.device)
            if config.refine_val != REFINE_VAL:
                raise RuntimeError('{} refines by {}, but this runner assumes {}'
                                   .format(config_name, config.refine_val, REFINE_VAL))
            ckpt, note = checkpoint_for(config_name, self.requested_ckpt)
            if note and config_name not in self._noted:
                print('note       : {}'.format(note))
                self._noted.add(config_name)
            self._loaded[config_name] = (config, load_network(ckpt, config, self.device))
        return self._loaded[config_name]


def strip_alpha(raw):
    """Return `raw` without its alpha channel, if it has one.

    OpenCV puts alpha last in both layouts it can hand back -- four channels as BGRA and
    two as grey+alpha -- so it comes off the end either way, and what is left is what
    upstream's channel reversal expects to see.  Upstream would reverse a two-channel last
    axis and swap grey with alpha, so those two are separated here rather than there.
    """
    if raw is not None and raw.ndim == 3:
        if raw.shape[2] == 4:
            return raw[:, :, :3]
        if raw.shape[2] == 2:
            return raw[:, :, 0]
    return raw


def read_input(path, config=None):
    """Read `path` as (1, h, w, 3) float RGB in [0, 1], with the bits per channel used.

    With a `config` given, the mode is fixed by it.  With `config=None` the file decides,
    which is what makes a 16-bit input come back 16-bit without being asked: formats that
    cannot hold more than 8 bits go straight to upstream's `read_frame`, and anything
    deeper is read at native depth.

    The 8-bit read is upstream's, untouched.  The 16-bit read is upstream's too, except
    that it takes an alpha channel off first -- upstream reverses the last axis, which
    would turn a four-channel BGRA image into ARGB and quietly corrupt it, and Affinity
    TIFF exports commonly carry alpha.  Dropping the alpha is the specification rather
    than a failure: the original file still has it, and the whole frame is what a
    correction layer wants.

    A DNG is read as a bare TIFF, since that is all OpenCV sees.  It says so rather than
    pretending: everything that makes a DNG a DNG is skipped.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.dng':
        print('WARNING    : {} is a DNG, read as a bare TIFF: its colour matrix, white '
              'balance and linearisation are skipped, so it will not match the same '
              'picture exported as TIFF or PNG'.format(os.path.basename(path)))

    raw = None
    if config is None:
        # The file's own depth decides.  For a format that is always 8 bits this needs no
        # read at all; otherwise the read below doubles as the probe, so nothing is read
        # twice.
        if ext in SHALLOW_EXTS:
            bits = 8
        else:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError('could not read {}'.format(path))
            bits = raw.dtype.itemsize * 8
    else:
        bits = getattr(config, 'in_bit', 8)

    if bits <= 8:
        return read_frame(path, 255, None), 8

    if raw is None:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError('could not read {}'.format(path))
    file_bits = raw.dtype.itemsize * 8
    if file_bits < 16:
        raise RuntimeError(
            '{} holds {}-bit data but this run reads 16-bit, and dividing it by 65535 '
            'would bring it back almost black; drop --config to let each file keep its '
            'own depth'.format(path, file_bits))
    if raw.dtype != np.uint16:
        raise RuntimeError(
            '{} holds {} data, and this runner reads 8- and 16-bit unsigned images only'
            .format(path, raw.dtype))

    raw = strip_alpha(raw)
    if raw.ndim == 2:
        print('WARNING    : {} is single-channel, so it is treated as greyscale; if this '
              'is a raw file it is a colour-filter array (a mosaic) and the result is '
              'meaningless'.format(os.path.basename(path)))
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    else:
        raw = raw[:, :, ::-1]                      # BGR -> RGB

    return np.expand_dims(raw / 65535.0, axis=0), 16


def to_output(blended, in_bit):
    """(h, w, 3) float RGB in [0, 1] -> BGR array at the config's bit depth."""
    scale = (2 ** in_bit) - 1
    dtype = np.uint16 if in_bit == 16 else np.uint8
    return (np.flip(blended, 2) * scale).astype(dtype)   # RGB -> BGR for OpenCV


def jpeg_chroma_sampling(path):
    """Return '444', '422' or '420' for a JPEG, or None for anything else.

    Read from the frame header directly rather than through an image library: it is the
    number the file was encoded with, and this runner otherwise depends on OpenCV alone.
    """
    try:
        with open(path, 'rb') as handle:
            data = handle.read(64 * 1024)
    except OSError:
        return None
    if data[:2] != b'\xff\xd8':
        return None
    position = 2
    while position + 12 <= len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            position += 2
            continue
        length = int.from_bytes(data[position + 2:position + 4], 'big')
        if marker in (0xC0, 0xC1, 0xC2, 0xC9, 0xCA, 0xCB):      # the frame headers
            if data[position + 9] < 3:
                return '444'                                    # no colour to subsample
            horizontal, vertical = data[position + 11] >> 4, data[position + 11] & 0x0F
            return {(1, 1): '444', (2, 1): '422', (2, 2): '420'}.get((horizontal, vertical))
        position += 2 + length
    return None


def jpeg_write_params(input_path):
    """Writer options for a JPEG result: keep the input's own chroma sampling.

    Nothing is gained by writing a better 4:4:4 result than the 4:2:0 the file arrived
    in, and something is lost by writing a worse one, so the result is neither: it
    matches.  OpenCV older than 4.5.5 cannot be told, and keeps its own 4:2:0 default.
    """
    if JPEG_SAMPLING is None:
        return []
    return [JPEG_SAMPLING, JPEG_SAMPLING_FACTORS[jpeg_chroma_sampling(input_path) or '444']]


def container_name(path):
    """The container as the interface contract names it, taken from the file name."""
    extension = os.path.splitext(path)[1].lower().lstrip('.')
    return {'jpg': 'jpeg', 'jpe': 'jpeg', 'tif': 'tiff'}.get(extension, extension)


def now_iso():
    """Local time with its UTC offset, the form the contract shows."""
    return datetime.now().astimezone().isoformat(timespec='seconds')


def result_entry(path, status, output=None, error=None, original=None, processed=None,
                 bits=None, seconds=None):
    """One `results` entry, in the shape INTERFACE.md fixes."""
    return {
        'input': os.path.abspath(path),
        'output': os.path.abspath(output) if output else None,
        'status': status,
        'error': error,
        'input_size': [original[0], original[1]] if original else None,
        'output_size': [processed[0], processed[1]] if processed else None,
        'offset': [0, 0] if status == 'ok' else None,
        'bit_depth': bits,
        'channels': 3 if status == 'ok' else None,
        'container': container_name(output) if output else None,
        'seconds': seconds,
    }


def write_manifest(path, manifest):
    """Write the run record, replacing any earlier copy in one step.

    Called once as the run starts and again as each file lands, so a run that is
    interrupted still leaves a record of what it managed to do rather than nothing.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + '.part'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def process(runner, device, path, output_path, max_side, tile, overlap, max_pixels,
            forced=None):
    """Deblur one image and write it to `output_path`.

    `forced` names a configuration to use whatever the file holds; with it left None the
    file's own bit depth decides, and the result is written at that depth.  The container
    is checked once the file has been read and before any inference, so a mismatch costs
    only the read.
    """
    config = net = None
    if forced:
        config, net = runner.get(forced)

    frame, bits = read_input(path, config)   # (1, h, w, 3) float RGB in [0, 1]
    _, height, width, _ = frame.shape

    if config is None:
        config, net = runner.get(config_name_for_bit_depth(bits))

    problem = container_error(bits, os.path.splitext(output_path)[1].lower())
    if problem:
        raise RuntimeError(problem)

    note = ''
    if max_side and max(height, width) > max_side:
        ratio = max_side / max(height, width)
        scaled = (int(width * ratio), int(height * ratio))
        frame = np.expand_dims(
            cv2.resize(frame[0], dsize=scaled, interpolation=cv2.INTER_AREA), 0)
        note = ' downscaled {}x{}->{}x{}'.format(width, height, scaled[0], scaled[1])

    # Past the ceiling we stop rather than quietly produce a smaller picture.  Both what
    # the image wants and what the limit is are named, so the answer to "why this number"
    # is on screen rather than in a table somewhere.
    pixels = frame.shape[1] * frame.shape[2]
    if max_pixels and pixels > max_pixels:
        raise RuntimeError(
            '{}x{} = {:.1f} megapixels is over the {:.1f} megapixel limit, and would '
            'want about {:.1f} GiB of RAM; raise the limit with --max-pixels if you '
            'know better'.format(frame.shape[2], frame.shape[1], pixels / 1e6,
                                 max_pixels / 1e6, image_ram(pixels) / 1024 ** 3))

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

    output = to_output(blended, config.in_bit)   # RGB -> BGR

    # OpenCV defaults TIFF to LZW, which *inflates* 16-bit output: 147 MiB for data
    # that is 119 MiB raw, because the low byte of a smooth 16-bit gradient is close
    # to random.  Adobe Deflate is lossless and gets the same picture into 108 MiB.
    compressed = (config.in_bit == 16
                  and output_path.lower().endswith(('.tif', '.tiff')))
    params = []
    if compressed:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, TIFF_DEFLATE]
    elif output_path.lower().endswith(('.jpg', '.jpeg')):
        params = jpeg_write_params(path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = cv2.imwrite(output_path, output, params)
    if not written:
        raise RuntimeError('could not write {}'.format(output_path))

    return elapsed, (width, height), (out_w, out_h), note, tiles, config.in_bit


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
    parser.add_argument('--ckpt', default=None,
                        help='path to the pretrained checkpoint (default: ckpt\\IFAN.pytorch, '
                             'or ckpt\\IFAN_16bit.pytorch for 16-bit input when it is there)')
    parser.add_argument('--max-side', type=int, default=DEFAULT_MAX_SIDE,
                        help='downscale inputs whose longest side exceeds this many pixels; '
                             '0 (the default) never downscales')
    parser.add_argument('--max-pixels', type=int, default=None,
                        help='refuse images above this many pixels instead of shrinking them; '
                             'the default is what this machine has the memory for, and '
                             '0 removes the ceiling')
    parser.add_argument('--tile', type=int, default=DEFAULT_TILE,
                        help='largest tile side when processing large images; '
                             '0 disables tiling and needs memory proportional to the image '
                             '(default: %(default)s)')
    parser.add_argument('--overlap', type=int, default=DEFAULT_OVERLAP,
                        help='feather width between tiles (default: %(default)s)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='recurse into subfolders, keeping the folder structure')
    parser.add_argument('--cpu', action='store_true', help='run on CPU even if a GPU is available')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        choices=[DEFAULT_CONFIG] + list(CONFIGS),
                        help='model configuration: auto (default) matches each input file\'s '
                             'own bit depth; config_IFAN forces 8-bit in and out; '
                             'config_IFAN_16bit forces 16-bit')
    parser.add_argument('--ext', default='keep', choices=['keep', 'png', 'jpg', 'tif'],
                        help='output format; "keep" preserves each input extension (default)')
    parser.add_argument('--manifest', default=None, metavar='PATH',
                        help='write a JSON record of the run (interface contract %d) to '
                             'PATH; a second program reads that rather than the console'
                             % CONTRACT)
    parser.add_argument('--version', action='version',
                        version='{} {}'.format(TOOL_ID, TOOL_VERSION),
                        help='print the tool id and version, "%s %s", and exit'
                             % (TOOL_ID, TOOL_VERSION))
    args = parser.parse_args()

    # Naming --output is shorthand for --dest output.
    if args.dest is None:
        args.dest = 'output' if args.output is not None else DEFAULT_DEST
    if args.dest == 'output' and args.output is None:
        args.output = os.path.join(HERE, 'output')

    if not os.path.exists(args.input):
        sys.exit('Input not found: {}'.format(args.input))
    if args.ckpt and not os.path.isfile(args.ckpt):
        sys.exit('Checkpoint not found: {}'.format(args.ckpt))

    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')

    tile = args.tile if args.tile <= 0 else args.tile - args.tile % REFINE_VAL
    if args.tile > 0 and tile != args.tile:
        print('note       : tile rounded down to {} (multiple of {})'.format(tile, REFINE_VAL))
    overlap = args.overlap - args.overlap % REFINE_VAL

    # The ceiling guards host RAM, so it is worked out from the machine at hand rather than
    # fixed in this file: the number that suits a 32 GB workstation does not suit a laptop,
    # and whoever is at a different machine needs to be able to see where it came from.
    # --max-pixels settles it either way.
    if args.max_pixels is None:
        ceiling = pixel_ceiling()
        if ceiling is None:
            args.max_pixels = DEFAULT_MAX_PIXELS
            limit_note = ' (default: this machine would not say how much memory it has)'
        else:
            args.max_pixels = ceiling
            limit_note = ' (about half of what is free, at ~{} bytes per pixel)'.format(
                BYTES_PER_PIXEL)
    else:
        limit_note = ' (as asked)'

    device_name = '{}{}'.format(device, '' if device.type == 'cpu' else
                                ' ({})'.format(torch.cuda.get_device_name(0)))
    print('device     : {}'.format(device_name))
    print('model      : {}'.format(args.config if args.config != DEFAULT_CONFIG else
                                   'auto, each file at its own bit depth'))
    reading = memory_reading()
    if reading:
        print('memory     : {:.1f} GiB free of {:.1f} GiB'.format(
            reading[0] / 1024 ** 3, reading[1] / 1024 ** 3))
    print('max side   : {}'.format('unlimited' if not args.max_side else args.max_side))
    print('tiling     : {}'.format('disabled' if tile <= 0 else
                                   '{} px tiles, {} px overlap'.format(tile, overlap)))
    print('pixel limit: {}{}'.format('none' if not args.max_pixels else
                                     '{:.0f} megapixels'.format(args.max_pixels / 1e6),
                                     limit_note))
    if args.manifest:
        print('manifest   : {}'.format(os.path.abspath(args.manifest)))

    # Configurations are loaded on demand, one per bit depth actually encountered, so all
    # that can be checked here is that there is a checkpoint to load at all.
    forced = None if args.config == DEFAULT_CONFIG else args.config
    if not args.ckpt and not any(os.path.isfile(p) for p in CKPTS.values()):
        sys.exit('No checkpoint found in {}\nDownload one from the README links and place '
                 'it there.'.format(os.path.dirname(FALLBACK_CKPT)))
    runner = Runner(device, args.ckpt)

    # Results are written relative to the input: into a subfolder of it, beside each
    # image with a suffix, or both.  "output" sends them to a separate root instead.
    root = args.input if os.path.isdir(args.input) else os.path.dirname(os.path.abspath(args.input))
    subfolder = os.path.join(root, args.subfolder)
    in_subfolder = args.dest in ('subfolder', 'both')
    beside = args.dest in ('suffix', 'both')

    # The run record promised by INTERFACE.md.  It carries the options the contract names
    # and no others: where a result went is the `output` path inside its own entry, so a
    # handler never has to reconstruct a path from the options it passed.
    manifest = None
    if args.manifest:
        manifest = {
            'contract': CONTRACT,
            'tool': {'id': TOOL_ID, 'version': TOOL_VERSION},
            'started': now_iso(),
            'finished': None,
            'device': device_name,
            'options': {
                'dest': args.dest,
                'suffix': args.suffix,
                'tile': tile,
                'overlap': overlap,
                'max_pixels': args.max_pixels,
                'config': args.config,
                'ext': args.ext,
            },
            'counts': {'found': 0, 'written': 0, 'failed': 0, 'skipped': 0},
            'results': [],
        }

    def finish(written=0, failed=0):
        """Stamp the record as finished and put it on disk, if one was asked for."""
        if not manifest:
            return
        manifest['counts'].update(written=written, failed=failed)
        manifest['finished'] = now_iso()
        write_manifest(args.manifest, manifest)

    images = find_images(args.input, args.recursive)
    # Never pick up our own earlier results, or a second run would deblur the
    # deblurred copies and the folder would grow on every run.
    skipped = 0
    skipped_paths = []
    if in_subfolder or beside:
        kept = []
        for path in images:
            if in_subfolder and is_within(path, subfolder):
                skipped += 1
                skipped_paths.append(path)
            elif (beside and args.suffix
                  and os.path.splitext(os.path.basename(path))[0].endswith(args.suffix)):
                skipped += 1
                skipped_paths.append(path)
            else:
                kept.append(path)
        images = kept

    if manifest:
        # Skipped files are listed rather than only counted, so a handler can tell which
        # ones already had a result without walking the folder itself.
        manifest['results'].extend(result_entry(path, 'skipped') for path in skipped_paths)
        manifest['counts'].update(found=len(images) + skipped, skipped=skipped)
        write_manifest(args.manifest, manifest)

    if not images:
        if skipped:
            print('Nothing to do: all {} image(s) under {} are already processed'
                  .format(skipped, args.input))
            finish()
            return
        finish()
        sys.exit('No images found under {}'.format(args.input))

    # With one mode forced on every file, what the run produces is known up front, so a
    # mismatch is reported before any work rather than one file at a time.
    if forced:
        forced_bits = 16 if forced == 'config_IFAN_16bit' else 8
        for path in images:
            output_ext = output_extension(os.path.splitext(path)[1].lower(), args.ext)[0]
            problem = container_error(forced_bits, output_ext)
            if problem:
                # A forced configuration that cannot be written is a whole-run mistake, so
                # the run stops -- but the file it stopped on is in the record, with the
                # reason, rather than leaving a handler to guess from the exit code.
                message = '{}: {}'.format(os.path.basename(path), problem)
                if manifest:
                    manifest['results'].append(result_entry(path, 'failed', error=message))
                    manifest['counts'].update(found=len(images), failed=1)
                finish(0, 1)
                sys.exit(message)
        if forced_bits == 8:
            probe = next((p for p in images
                          if os.path.splitext(p)[1].lower() in DEEP_EXTS
                          and (input_bit_depth(p) or 8) > 8), None)
            if probe:
                print('WARNING    : {} holds 16-bit data but config_IFAN reads and writes '
                      '8-bit, so the extra precision is discarded; drop --config to keep '
                      'each file at its own depth'.format(os.path.basename(probe)))

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
    substitution_noted = False
    for index, path in enumerate(images, 1):
        relative = os.path.relpath(path, root)
        directory, filename = os.path.split(relative)
        stem, input_ext = os.path.splitext(filename)
        ext, changed = output_extension(input_ext, args.ext)
        if changed and not substitution_noted:
            print('note       : {} cannot be written back out, so results are TIFF'
                  .format(input_ext))
            substitution_noted = True
        name = stem + (args.suffix if beside else '') + ext
        if args.dest == 'suffix':
            output_path = os.path.join(root, directory, name)
        else:
            output_path = os.path.join(subfolder if in_subfolder else args.output,
                                       directory, name)

        try:
            elapsed, original, processed, note, tiles, bits = process(
                runner, device, path, output_path,
                args.max_side, tile, overlap, args.max_pixels, forced)
        except Exception as ex:
            failed += 1
            print('[{:>3}/{}] FAILED {}: {}'.format(index, len(images), relative, ex))
            if manifest:
                manifest['results'].append(result_entry(path, 'failed', error=str(ex)))
                manifest['counts']['failed'] = failed
                write_manifest(args.manifest, manifest)
            continue

        done += 1
        total += elapsed
        if device.type == 'cuda':
            peak = ' peak GPU {:.0f} MiB'.format(torch.cuda.max_memory_allocated() / 1024 ** 2)
        else:
            peak = ''
        tiled = ' [{} tiles]'.format(tiles) if tiles > 1 else ''
        print('[{:>3}/{}] {}  {}x{} -> {}x{}  {}-bit{}{}  {:.2f}s{}'
              .format(index, len(images), relative, original[0], original[1],
                      processed[0], processed[1], bits, note, tiled, elapsed, peak))

        if manifest:
            manifest['results'].append(result_entry(
                path, 'ok', output=output_path, original=original, processed=processed,
                bits=bits, seconds=round(elapsed, 2)))
            manifest['counts']['written'] = done
            write_manifest(args.manifest, manifest)

    print('')
    summary = 'Done: {} of {} image(s) in {:.1f}s'.format(done, len(images), total)
    if done:
        summary += ' (average {:.2f}s)'.format(total / done)
    if failed:
        summary += ', {} failed'.format(failed)
    print(summary)

    finish(done, failed)
    # The contract's exit code: 0 only when every requested file was written.
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
