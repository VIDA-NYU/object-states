import os
from PIL import Image
import supervision as sv
# from IPython import embed
from .eta_format import *
from .color import *
from .video import crop_box_with_size
from .step_annotations import load_object_annotations, get_obj_ann
from ..config import get_cfg
from IPython import embed


RESOURCES_DIR = '/datasets'

eta_dataset = '/datasets/annotation_final'
export_dataset = '/datasets/annotation_final_imagenet'
name = 'objstate_imagenet'


from functools import lru_cache
@lru_cache(50)
def warn_once(*a, **kw):
    log.warning(*a, **kw)


# import ipdb
# @ipdb.iex
# def main(
#         dataset_dir, 
#         export_dataset_dir=None, 
#         detections_field='detections_tracker', 
#         max_fps=3,
#     ):
#     print(dataset_dir, export_dataset_dir)
#     dataset = fo.Dataset.from_dir(
#         dataset_dir=dataset_dir,
#         dataset_type=fo.types.FiftyOneVideoLabelsDataset)

#     dataset = dataset.to_frames(sample_frames=True, max_fps=max_fps)
#     dataset = dataset.to_patches(detections_field)
    
#     frames.export(
#         export_dir=export_dir,
#         dataset_type=fo.types.ImageClassificationDirectoryTree,
#         label_field=detections_field,
#     )



def extract_frames(
    output_dir, *dataset_dirs, state_col='state', val_dataset_dirs=None, 
    train_keywords=None, val_keywords=None, fps_skip=10, 
    overwrite=False, always_iter_video=False,
    cfg='config/eval_dino_imagenet.yaml'
):
    cfg = get_cfg(cfg)
    dfs = load_object_annotations(cfg)

    output_dir = os.path.join(output_dir, state_col)

    train_keywords = [x for x in (train_keywords.split(',') if isinstance(train_keywords, str) else train_keywords or []) if x]
    val_keywords = [x for x in (val_keywords.split(',') if isinstance(val_keywords, str) else val_keywords or []) if x]
    dataset_dirs = list(dataset_dirs or [])
    val_dataset_dirs = list(val_dataset_dirs or [])

    for dataset_dir in dataset_dirs + val_dataset_dirs:
        manifest = load(manifest_fname(dataset_dir))
        pbar = tqdm.tqdm(sorted(manifest['index'], key=lambda d: d['labels']))
        for m in pbar:
            label_fname = m['labels']
            name = os.path.splitext(os.path.basename(label_fname))[0]
            # get dataset split
            if dataset_dir in val_dataset_dirs:
                split = 'train' if any(k in label_fname for k in train_keywords) else 'val'
            else:
                split = 'val' if any(k in label_fname for k in val_keywords) else 'train'

            pbar.set_description(name)
            ddf = dfs.get(name)

            # check if we actually need to load the video
            if not ddf:
                log.warning(f"{red('No tracks')} {split} {name}")
                continue
            if overwrite is False:
                fs = glob.glob(f'{output_dir}/*/*/{name}__*.JPEG')
                if fs:
                    # check if files are in another split. if yes, move them
                    files_to_move = []
                    for f in fs:
                        fparts = f.rsplit('/', 4)
                        if fparts[-3] != split:
                            fparts[-3] = split
                            files_to_move.append((f, '/'.join(fparts)))
                    if files_to_move:
                        log.warning(f"{yellow('** Renaming')}: %s files to %s (e.g. \n  %s)", len(files_to_move), split, files_to_move[0])
                        for f, f2 in files_to_move:
                            os.makedirs(os.path.dirname(f2) or '.', exist_ok=True)
                            os.rename(f, f2)
                    
                    if not always_iter_video:
                        log.warning(f"{yellow('Skipping')} {split} {name}")
                        continue
            log.warning(f"{green('Doing')} {split} {name}")

            # ok we need to load the video and annotations
            labels = load(label_fname)
            # if name == 'tea_2023.06.30-19.29.16':
            #     embed()
            for i, frame in enumerate(sv.get_video_frames_generator(source_path=m['data'])):
                if fps_skip and i % fps_skip: continue

                for o in get_objects(labels, i):
                    # object track ID
                    idx = o['index']

                    # check if we have state annotations
                    odf = ddf.get(idx)
                    if odf is None:
                        warn_once(f"{red('missing')} track state %s %s", name, idx)
                        continue
                    warn_once(f"using track index %s %s", name, idx)

                    # get object state and label
                    label, state = get_obj_ann(odf, i, state_col)

                    # check if file already exists
                    fname = f'{output_dir}/{split}/{label}__{state}/{name}__{i}__{idx}.JPEG' # extension imagenet expects
                    if os.path.isfile(fname) and not overwrite:
                        continue

                    warn_once(f"{green('extracting')} track state %s %s", name, idx)
                    pbar.set_description(f'{split} {label} {state} - {os.path.basename(fname)}')

                    # get cropped image
                    try:
                        xyxy = box_to_xyxy(o['bounding_box'], frame.shape[:2][::-1])
                        crop = crop_box_with_size(frame, xyxy, (224, 224), padding=15)
                    except Exception as e:
                        log.exception(e)
                        continue
                    im = Image.fromarray(crop[:,:,::-1])

                    # write to file
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    im.save(fname)

    dump_extra(output_dir)
    describe(output_dir)


def dump_extra(dataset_dir):
    import csv
    import pathtrees as pt
    image_path = pt.Path(dataset_dir) / '{split}/{label}/{fname}.JPEG'
    labels = sorted({image_path.parse(f)['label'] for f in image_path.glob()})
    print(labels)
    
    with open(os.path.join(dataset_dir, 'labels.txt'), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows([(l,l) for l in labels])
    
    
    from dinov2.data.datasets import ImageNet
    for split in ImageNet.Split:
        dataset = ImageNet(split=split, root=dataset_dir, extra=dataset_dir)
        dataset.dump_extra()


def describe(dataset_dir):
    import pandas as pd
    import pathtrees as pt
    image_path = pt.Path(dataset_dir) / '{split}/{label}/{name}__{i}__{idx}.JPEG'
    df = pd.DataFrame([
        image_path.parse(im)
        for im in image_path.glob()
    ])
    df = df[~df.split.str.startswith('_')]
    print(df.shape)
    print(df.head())
    # Print the pivot_table
    print(df.groupby(['split', 'label']).size().unstack().fillna(0).T.astype(int))








if __name__ == '__main__':
    from tqdm.contrib.logging import logging_redirect_tqdm
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        import fire
        fire.Fire()