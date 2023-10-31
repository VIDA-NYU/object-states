import io
import zlib
from base64 import b64decode
import numpy as np

# in deserialize_numpy_array(numpy_str, allow_pickle)
#     282 """Loads a serialized numpy array from string.
#     283 
#     284 Args:
#    (...)
#     290     the numpy array
#     291 """
#     292 #
#     293 # We currently serialize in numpy format. Other alternatives considered
#     294 # were `pickle.loads(numpy_str)` and HDF5
#     295 #
# --> 296 bytes_str = zlib.decompress(b64decode(numpy_str.encode("ascii")))
#     297 with io.BytesIO(bytes_str) as f:
#     298     return np.load(f, allow_pickle=allow_pickle)
# AttributeError: 'list' object has no attribute 'encode'
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