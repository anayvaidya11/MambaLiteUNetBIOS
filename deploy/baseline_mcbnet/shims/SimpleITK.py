"""Minimal SimpleITK stand-in for hybridTRTinf_plot.py: ReadImage + GetArrayFromImage for
NRRD files (raw or gzip encoding, attached data), returning arrays in SimpleITK's (Z, Y, X)
order. Lets Peter's script run in a venv without the real package; nothing in the timed
window touches it."""
import gzip

import numpy as np

_TYPES = {"unsigned char": np.uint8, "uchar": np.uint8, "uint8": np.uint8, "uint8_t": np.uint8,
          "signed char": np.int8, "int8": np.int8, "int8_t": np.int8,
          "short": np.int16, "int16": np.int16, "int16_t": np.int16,
          "unsigned short": np.uint16, "ushort": np.uint16, "uint16": np.uint16, "uint16_t": np.uint16,
          "int": np.int32, "int32": np.int32, "int32_t": np.int32,
          "unsigned int": np.uint32, "uint": np.uint32, "uint32": np.uint32, "uint32_t": np.uint32,
          "long long": np.int64, "int64": np.int64, "int64_t": np.int64,
          "unsigned long long": np.uint64, "uint64": np.uint64, "uint64_t": np.uint64,
          "float": np.float32, "double": np.float64}


class Image:
    def __init__(self, array):
        self._array = array

    def GetSize(self):  # noqa: N802  (SimpleITK naming)
        return tuple(int(s) for s in self._array.shape[::-1])


def ReadImage(path):  # noqa: N802
    with open(path, "rb") as f:
        if not f.readline().startswith(b"NRRD"):
            raise ValueError(f"{path}: not an NRRD file")
        header = {}
        while True:
            line = f.readline()
            if not line or line in (b"\n", b"\r\n"):
                break
            if line.startswith(b"#"):
                continue
            key, _, val = line.decode("utf-8", "replace").partition(":")
            header[key.strip().lower()] = val.strip()
        data = f.read()
    if "data file" in header or "datafile" in header:
        raise NotImplementedError(f"{path}: detached NRRD data is not supported")
    dtype = np.dtype(_TYPES[header["type"]])
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if header.get("endian", "little") == "little" else ">")
    enc = header.get("encoding", "raw").lower()
    if enc in ("gzip", "gz"):
        data = gzip.decompress(data)
    elif enc != "raw":
        raise NotImplementedError(f"{path}: NRRD encoding {enc!r} is not supported")
    sizes = [int(s) for s in header["sizes"].split()]
    arr = np.frombuffer(data, dtype=dtype, count=int(np.prod(sizes))).reshape(sizes, order="F")
    return Image(np.ascontiguousarray(arr.T))          # (X, Y, Z) -> (Z, Y, X)


def GetArrayFromImage(image):  # noqa: N802
    return image._array
