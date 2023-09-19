from . import _patch

import cv2
import numpy as np
import fiftyone as fo
import supervision as sv
from .util.video import XMemSink
from .util.format_convert import fo_to_sv

import ipdb
@ipdb.iex
def main(data_dir='/datasets/export1', fields='detections_tracker'):
    
    # ------------------------------- Load dataset ------------------------------- #

    dataset = fo.Dataset.from_dir(
        dataset_dir=data_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
    )

    view = dataset.view()
    view.compute_metadata(overwrite=True)

    for sample in tqdm.tqdm(view):
        with XMemSink(out_path, video_info) as s:
            for i, frame, finfo in iter_video(sample):
                # get detection
                detections, labels = fo_to_sv(finfo[field])
                s.write_frame(frame)


if __name__ == '__main__':
    import fire
    fire.Fire(main)