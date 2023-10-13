import orjson


def json_dump(fname, data):
    with open(fname, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY))
