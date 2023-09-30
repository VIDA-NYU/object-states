import os
import fiftyone as fo
from IPython import embed
from object_states import _patch


def convert_to_coco(dataset_dir, export_dir):
    # load
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset,
    )

    frames = dataset.to_frames(sample_frames=True).clone()

    # Export the dataset
    frames.export(
        export_dir=export_dir,
        dataset_type=fo.types.COCODetectionDataset,
        label_field='ground_truth_tracker',
    )

def main(*dataset_dirs, out_dir='/datasets/coco_export'):
    for dataset_dir in dataset_dirs:
        convert_to_coco(dataset_dir, os.path.join(out_dir, os.path.basename(dataset_dir)))

if __name__ == '__main__':
    import fire
    fire.Fire(main)