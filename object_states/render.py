import os
import tqdm
import fiftyone as fo
import supervision as sv
from .util.video import XMemSink, iter_video
from .util.format_convert import fo_to_sv

import ipdb
@ipdb.iex
def main(dataset_dir, field='detections_tracker'):
    
    # ------------------------------- Load dataset ------------------------------- #

    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
    )

    view = dataset.view()
    view.compute_metadata(overwrite=True)

    for sample in tqdm.tqdm(view):

        video_path = sample.filepath
        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        out_path = f'{dataset_dir}/track_render/{os.path.basename(video_path)}.mp4'

        with XMemSink(out_path, video_info) as s:
            for i, frame, finfo in iter_video(sample):
                detections, labels = fo_to_sv(finfo[field])
                s.write_frame(frame, detections, labels)


if __name__ == '__main__':
    import fire
    fire.Fire(main)