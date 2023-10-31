import os

# ---------------------------------------------------------------------------- #
#                             Available Databases:                             #
# ---------------------------------------------------------------------------- #

# https://drive.google.com/drive/folders/1AxIWbVOyFfu8agILKyLTDNN7uJoK9A8f
DEFAULT_KEY = 'v0'
FILE_IDS = {
    "v0": "10xIr2Wx07GqGm79VTAPh6NrYQgRlehGy",
    "v1": "",
}


# ---------------------------------------------------------------------------- #
#                                  Downloader                                  #
# ---------------------------------------------------------------------------- #

BASE_URL = 'https://docs.google.com/uc?export=download&confirm=t&id={}'

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.getenv("MODEL_DIR") or os.path.join(ROOT_DIR, 'state_dbs')
EXT = '.lancedb'

def ensure_db(key=None, path=None, file_id=None, output_dir=OUTPUT_DIR, ext=EXT):
    '''Download a lancedb database zip file from Google Drive if it does not already exist.
    
    Arguments:
        key (str): The database key. See ``FILE_IDS`` for available options.
        path (str): The path to download the file to. By default: ``{output_dir}/{key}{ext}``
        file_id (str): Override the google drive file ID.
        output_dir (str): The output directory of the default path.
        ext (str): The extension to add to the key in the default path.

    Returns:
        path (str): The path that the file was downloaded to.
    '''
    key = key or DEFAULT_KEY
    if key and key not in FILE_IDS:
        # they passed an existing directory, trust that they want to use that.
        if os.path.isdir(key):
            return key

    file_id = file_id or FILE_IDS[key]
    path = path or os.path.join(output_dir or '.', f'{key}{ext}')
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        print(f"No checkpoint found. Downloading to {path}...")
        def show_progress(i, size, total):
            print(f'downloading checkpoint to {path}: {i * size / total:.2%}', end="\r")
        
        # download and unzip
        import urllib.request
        zip_path = f'{path}.zip'
        urllib.request.urlretrieve(BASE_URL.format(file_id), zip_path, show_progress)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(path)
        os.remove(zip_path)
    return path

