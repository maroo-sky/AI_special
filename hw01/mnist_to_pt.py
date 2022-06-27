import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
import gzip
import lzma
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path: Union[str, IO]) -> Union[IO, gzip.GzipFile]:
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        return lzma.open(path, 'rb')
    return open(path, 'rb')


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}


def read_sn3_pascalvincent_tensor(path: Union[str, IO], strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def convert_training_set(raw_folder, processed_folder, training_file, test_file):
    training_set = (
                read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte.gz')),
                read_label_file(os.path.join(raw_folder, 'train-labels-idx1-ubyte.gz'))
            )
    test_set = (
        read_image_file(os.path.join(raw_folder, 't10k-images-idx3-ubyte.gz')),
        read_label_file(os.path.join(raw_folder, 't10k-labels-idx1-ubyte.gz'))
    )
    with open(os.path.join(processed_folder, training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(os.path.join(processed_folder, test_file), 'wb') as f:
        torch.save(test_set, f)

def main():
    raw_folder = r"C:\Users\USER\Desktop\내꺼\GIST\석박통합\석사3학기\인공지능 특론1\hw01\dataset\MNIST\raw"
    processed_folder = r"C:\Users\USER\Desktop\내꺼\GIST\석박통합\석사3학기\인공지능 특론1\hw01\dataset\MNIST\processed"
    training_file = r"C:\Users\USER\Desktop\내꺼\GIST\석박통합\석사3학기\인공지능 특론1\hw01\dataset\MNIST\processed\training.pt"
    test_file = r"C:\Users\USER\Desktop\내꺼\GIST\석박통합\석사3학기\인공지능 특론1\hw01\dataset\MNIST\processed\test.pt"

    convert_training_set(raw_folder, processed_folder, training_file, test_file)

if __name__ == "__main__":
    main()
