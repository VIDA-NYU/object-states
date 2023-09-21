import io
import zlib
from base64 import b64decode
import numpy as np

# list has no attribute 'encode'
import eta.core.serial
def deserialize_numpy_array(numpy_str, allow_pickle=False):
    if isinstance(numpy_str, list):
        return np.array(numpy_str)
    bytes_str = zlib.decompress(b64decode(numpy_str.encode("ascii")))
    with io.BytesIO(bytes_str) as f:
        return np.load(f, allow_pickle=allow_pickle)
eta.core.serial.deserialize_numpy_array = deserialize_numpy_array


# ValidationError(ValidationError (frames.samples.6509fbcf050c5d6df85beb24:None) (Field is required: ['id']),)
from fiftyone.core.frame import FrameView
old_save = FrameView.save
def save(self, *a, **kw):
    self._fields['id'].required = False
    return old_save(self, *a, **kw)
FrameView.save = save