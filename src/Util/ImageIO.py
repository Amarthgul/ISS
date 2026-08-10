
import os
import OpenEXR
import struct
from datetime import datetime
from enum import Enum
from fractions import Fraction

import numpy as np


from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Misc import NumpyConversion, RectPath
from Util.Globals import RNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, TWO, LambdaLines, RefreshRNG, Axis, UP_DIR, ORIGIN, NEAR_ZERO


class SaveFormat(Enum):
    EXR = 0
    DNG = 1


def ImageConversion(ary, bitDepth=8, maxModifier=1, normalizer=None, rotate=True, flipH=False, flipV=False):
    """
    Convert the float representation of an image to an uint8 image.
    """
    bits = 2.0**bitDepth-1
    scaleBase = (ONE if normalizer is None else normalizer) * maxModifier
    scaleRatio = bits / (scaleBase + NEAR_ZERO)
    ary = bd.clip(ary*scaleRatio, 0, bits) 

    if rotate:
        ary = bd.rot90(ary)

    if flipH:
        ary = bd.flip(ary, axis=1)
    if flipV:
        ary = bd.flip(ary, axis=0)

    return NumpyConversion(ary).astype(bd.uint8)


def CleanDisplay(rgbArray):
    import matplotlib.pyplot as plt

    plt.close('all')  # kill all existing figures (optional but safe)
    fig = plt.figure()
    ax = fig.add_subplot(111)  # IMPORTANT: no projection='3d'
    plt.imshow(rgbArray)


def rgbFromRGBA(rgba: bd.ndarray, background=(255, 255, 255)):
    """
    Convert an RGBA image array (H, W, 4) to an RGB array (H, W, 3),
    compositing transparency over a solid background color.

    :param rgba: bd.ndarray input image, dtype uint8, shape (H, W, 4).
    :param background: Background color (R, G, B), default black.

    :return: RGB image, dtype float32 normalized to [0,1] (ready for plt.imshow()).
    """

    # Apparently, cupy does not do well with some of these operations, I had to switch it to full numpy.
    # But well, if someone is using this, it's probably not a
    import numpy as np

    rgba = np.asarray(rgba)  # ensure NumPy
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    bg = np.array(background, dtype=np.float32).reshape(1, 1, 3) / 255.0

    out = alpha * rgb + (1.0 - alpha) * bg

    return out


def ImageConversionAverage(ary, bitDepth=8, modifier=2, rotate=True):
    """
    Convert the float representation of an image to a uint8 image.
    """
    emanVal = bd.median(ary)

    bits = 2.0**bitDepth-1
    scaleRatio = bits / (emanVal * modifier)
    ary = bd.clip(ary*scaleRatio, 0, bits) 

    if(rotate):
        ary = bd.rot90(ary)

    return NumpyConversion(ary).astype(bd.uint8)


def SaveAsEXR(ary, folder, fileName, *extra_channels, flipHori=False, flipVert=False, rotate=False):
    """
    Save the latent image array as an EXR file.

    :param ary: base image array. This will automatically be converted to numpy float32.
    :param folder: folder location to save the image, note that the folder must exist.
    :param fileName: the whole file name (without extension).
    :param extra_channels: optional extra channels, passed as (array, name) pairs, e.g.:

        SaveAsEXR(rgb, "output", "image",
                  depth_array, "Z",
                  mask_array, "MASK")

        Each array will be saved as a separate EXR channel with the given name.
    :param flipHori: flip the whole image horizontally (left-right mirror).
    :param flipVert: flip the whole image vertically (top-bottom mirror).
    """

    def _flip_image_orientation(a):
        if a.ndim < 2:
            return a

        if rotate:
            a = bd.rot90(a, k=1)

        # Image-space convention:
        # axis 0 = vertical direction   (top-bottom)
        # axis 1 = horizontal direction (left-right)
        if flipVert:
            a = bd.flip(a, axis=0)
        if flipHori:
            a = bd.flip(a, axis=1)

        return a

    def _to_numpy_f32(a):
        a = a.astype(bd.float32)
        a = _flip_image_orientation(a)

        if backend_name == 'cupy':
            a = bd.asnumpy(a)
        return a

    ary = bd.asarray(ary)
    if ary.ndim != 3 or ary.shape[-1] < 3:
        raise ValueError(f"SaveAsEXR expects base image to have shape (H, W, C>=3). Got {ary.shape}")

    # Base RGB channel. Any latent AOVs carried after RGB are intentionally not
    # saved here; callers must pass output channels explicitly.
    RGB = _to_numpy_f32(ary[:, :, :3])

    channels = {"RGB": RGB}

    # Parse extra channels: expect (array, name) pairs
    if extra_channels:
        for arr, ch_name in zip(extra_channels[0::2], extra_channels[1::2]):
            channels[ch_name] = _to_numpy_f32(arr)

    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }

    folder = RectPath(folder)
    if folder[-1] != r"/":
        folder += r"/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    nameStr = folder + fileName + ".exr"

    with OpenEXR.File(header, channels) as outfile:
        outfile.write(nameStr)


# This product includes DNG technology under license by Adobe.
def SaveAsDNG(
        rawArray,
        folder,
        fileName,
        cfaPattern="RGGB",
        bitDepth=14,
        blackLevel=0,
        whiteLevel=None,
        metadata=None):
    """Save a normalized, row-major Bayer mosaic as an uncompressed DNG.

    Samples are stored in a 16-bit TIFF container. ``bitDepth`` and
    ``whiteLevel`` describe the simulated sensor ADC; using a 16-bit container
    avoids the non-byte-aligned packing required for native 10/12/14-bit TIFF
    samples.
    """
    raw = np.asarray(NumpyConversion(rawArray))
    if raw.ndim != 2:
        raise ValueError(f"SaveAsDNG expects a 2D Bayer mosaic. Got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("SaveAsDNG cannot encode non-finite raw values.")

    bitDepth = int(bitDepth)
    if bitDepth < 8 or bitDepth > 16:
        raise ValueError("SaveAsDNG bitDepth must be between 8 and 16.")

    blackLevel = int(blackLevel)
    if whiteLevel is None:
        whiteLevel = (1 << bitDepth) - 1
    whiteLevel = int(whiteLevel)
    if blackLevel < 0 or whiteLevel <= blackLevel or whiteLevel > 65535:
        raise ValueError(
            "SaveAsDNG requires 0 <= blackLevel < whiteLevel <= 65535."
        )

    pattern = str(cfaPattern).upper()
    if (
        len(pattern) != 4 or
        pattern.count("R") != 1 or
        pattern.count("G") != 2 or
        pattern.count("B") != 1
    ):
        raise ValueError("SaveAsDNG requires a 2x2 Bayer CFA pattern.")

    metadata = dict(metadata or {})
    encoded = np.rint(
        np.clip(raw.astype(np.float64), 0.0, 1.0) *
        (whiteLevel - blackLevel) + blackLevel
    ).astype("<u2")

    height, width = encoded.shape
    rawBytes = encoded.tobytes(order="C")

    BYTE = 1
    ASCII = 2
    SHORT = 3
    LONG = 4
    RATIONAL = 5
    SRATIONAL = 10

    def _ascii(value):
        return str(value).encode("utf-8") + b"\x00"

    def _short(values):
        values = tuple(int(value) for value in values)
        return struct.pack("<" + "H" * len(values), *values)

    def _long(values):
        values = tuple(int(value) for value in values)
        return struct.pack("<" + "I" * len(values), *values)

    def _rational(values, signed=False):
        packed = bytearray()
        code = "ii" if signed else "II"
        for value in values:
            fraction = Fraction(float(value)).limit_denominator(1_000_000)
            packed.extend(struct.pack("<" + code, fraction.numerator, fraction.denominator))
        return bytes(packed)

    def _entry(tag, fieldType, count, data):
        return [int(tag), int(fieldType), int(count), bytes(data)]

    channelID = {"R": 0, "G": 1, "B": 2}
    cfaValues = bytes(channelID[channel] for channel in pattern)
    colorMatrix = tuple(metadata.get("colorMatrix1", (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )))
    if len(colorMatrix) != 9:
        raise ValueError("DNG colorMatrix1 must contain nine values.")

    asShotNeutral = tuple(metadata.get("asShotNeutral", (1.0, 1.0, 1.0)))
    if len(asShotNeutral) != 3:
        raise ValueError("DNG asShotNeutral must contain three values.")

    make = metadata.get("make", "ISS")
    model = metadata.get("model", "Virtual PDA")
    uniqueModel = metadata.get("uniqueCameraModel", f"{make} {model}")
    software = metadata.get("software", "ISS Optical Simulation")
    dateTime = metadata.get("dateTime", datetime.now().strftime("%Y:%m:%d %H:%M:%S"))
    orientation = int(metadata.get("orientation", 1))
    if orientation < 1 or orientation > 8:
        raise ValueError("DNG orientation must be between 1 and 8.")

    entries = [
        _entry(254, LONG, 1, _long((0,))),
        _entry(256, LONG, 1, _long((width,))),
        _entry(257, LONG, 1, _long((height,))),
        _entry(258, SHORT, 1, _short((16,))),
        _entry(259, SHORT, 1, _short((1,))),
        _entry(262, SHORT, 1, _short((32803,))),
        _entry(271, ASCII, len(_ascii(make)), _ascii(make)),
        _entry(272, ASCII, len(_ascii(model)), _ascii(model)),
        _entry(273, LONG, 1, _long((0,))),
        _entry(274, SHORT, 1, _short((orientation,))),
        _entry(277, SHORT, 1, _short((1,))),
        _entry(278, LONG, 1, _long((height,))),
        _entry(279, LONG, 1, _long((len(rawBytes),))),
        _entry(284, SHORT, 1, _short((1,))),
        _entry(305, ASCII, len(_ascii(software)), _ascii(software)),
        _entry(306, ASCII, len(_ascii(dateTime)), _ascii(dateTime)),
        _entry(339, SHORT, 1, _short((1,))),
        _entry(33421, SHORT, 2, _short((2, 2))),
        _entry(33422, BYTE, 4, cfaValues),
        _entry(50706, BYTE, 4, bytes((1, 4, 0, 0))),
        _entry(50707, BYTE, 4, bytes((1, 1, 0, 0))),
        _entry(50708, ASCII, len(_ascii(uniqueModel)), _ascii(uniqueModel)),
        _entry(50710, BYTE, 3, bytes((0, 1, 2))),
        _entry(50711, SHORT, 1, _short((1,))),
        _entry(50713, SHORT, 2, _short((1, 1))),
        _entry(50714, RATIONAL, 1, _rational((blackLevel,))),
        _entry(50717, LONG, 1, _long((whiteLevel,))),
        _entry(50718, RATIONAL, 2, _rational((1.0, 1.0))),
        _entry(50719, LONG, 2, _long((0, 0))),
        _entry(50720, LONG, 2, _long((width, height))),
        _entry(50721, SRATIONAL, 9, _rational(colorMatrix, signed=True)),
        _entry(50728, RATIONAL, 3, _rational(asShotNeutral)),
        _entry(50730, SRATIONAL, 1, _rational((metadata.get("baselineExposure", 0.0),), signed=True)),
        _entry(50733, LONG, 1, _long((0,))),
        _entry(50778, SHORT, 1, _short((metadata.get("calibrationIlluminant1", 21),))),
        _entry(50829, LONG, 4, _long((0, 0, height, width)))
    ]

    iso = metadata.get("iso")
    if iso is not None:
        entries.append(_entry(34855, SHORT, 1, _short((iso,))))

    entries.sort(key=lambda item: item[0])
    ifdOffset = 8
    externalOffset = ifdOffset + 2 + 12 * len(entries) + 4
    externalLocations = {}

    for index, entry in enumerate(entries):
        data = entry[3]
        if len(data) <= 4:
            continue
        if externalOffset % 2:
            externalOffset += 1
        externalLocations[index] = externalOffset
        externalOffset += len(data)

    pixelOffset = (externalOffset + 3) & ~3
    for entry in entries:
        if entry[0] == 273:
            entry[3] = _long((pixelOffset,))
            break

    header = bytearray()
    header.extend(b"II")
    header.extend(struct.pack("<H", 42))
    header.extend(struct.pack("<I", ifdOffset))
    header.extend(struct.pack("<H", len(entries)))

    for index, (tag, fieldType, count, data) in enumerate(entries):
        header.extend(struct.pack("<HHI", tag, fieldType, count))
        if len(data) <= 4:
            header.extend(data.ljust(4, b"\x00"))
        else:
            header.extend(struct.pack("<I", externalLocations[index]))

    header.extend(struct.pack("<I", 0))

    for index, entry in enumerate(entries):
        data = entry[3]
        if len(data) <= 4:
            continue
        targetOffset = externalLocations[index]
        header.extend(b"\x00" * (targetOffset - len(header)))
        header.extend(data)

    header.extend(b"\x00" * (pixelOffset - len(header)))

    folder = RectPath(folder)
    os.makedirs(folder, exist_ok=True)
    outputPath = os.path.join(folder, fileName + ".dng")
    with open(outputPath, "wb") as outputFile:
        outputFile.write(header)
        outputFile.write(rawBytes)

    return outputPath


