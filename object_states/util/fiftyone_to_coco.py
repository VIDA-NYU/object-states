import os
import fiftyone as fo
from IPython import embed
from object_states import _patch


def convert_to_coco(*dataset_dirs, export_dir='/datasets/coco_export'):
    print("Source:", dataset_dirs)
    print("Destination:", export_dir)
    # load

    dataset = fo.Dataset()
    for dataset_dir in dataset_dirs:
        print(dataset_dir)
        d = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.FiftyOneVideoLabelsDataset,
        )
        dataset.merge_samples(d)
        # for sample in d:
        #     video_id = os.path.splitext(os.path.basename(sample.filepath))[0]
        #     if video_id == 'main':  # sometimes the videos are called main.mp4 so use the directory
        #         assert len(d) == 1, "hmm something weird is going on here..."
        #         video_id = os.path.basename(dataset_dir.rstrip('/'))
        #     sample = sample.to_frames(sample_frames=True, frames_patt=f"{video_id}_%%06d.jpg")
        #     print(video_id)
        #     input()
        #     dataset.add_sample(sample)

    # frames_patt = os.path.join(FRAMES_DIR, name, "%06d.jpg")
    # frames = dataset.to_frames(sample_frames=True)

    return 
    # Export the dataset
    frames.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field='ground_truth_tracker',
    )


if __name__ == '__main__':
    import fire
    fire.Fire(convert_to_coco)