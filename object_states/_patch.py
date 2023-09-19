import eta.core.serial
def deserialize_numpy_array(numpy_str, allow_pickle=False):
    if isinstance(numpy_str, list):
        return np.array(numpy_str)
    bytes_str = zlib.decompress(b64decode(numpy_str.encode("ascii")))
    with io.BytesIO(bytes_str) as f:
        return np.load(f, allow_pickle=allow_pickle)
eta.core.serial.deserialize_numpy_array = deserialize_numpy_array