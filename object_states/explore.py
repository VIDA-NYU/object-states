import os
from . import _patch
import fiftyone as fo
import fiftyone.brain as fob
from IPython import embed


RESOURCES_DIR = '/datasets'

eta_dataset = '/datasets/Milly'
export_dataset = '/datasets/Milly/object_labels'
name = 'milly'
max_samples=None

import ipdb
@ipdb.iex
def main(
        dataset_dir, 
        export_dataset_dir=None, 
        detections_field='detections_tracker', 
        model="clip-vit-base32-torch",
        proj='umap',
        n_unique=1000,
        batch_size=256
    ):
    video_dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneVideoLabelsDataset)

    frames_dataset = video_dataset.to_frames(sample_frames=True)
    sim_idx = fob.compute_similarity(
        frames_dataset, 
        patches_field=detections_field, 
        brain_key="emb_sim",
        model=model,
        batch_size=batch_size, 
        backend='lancedb',
        uri=os.path.join(dataset_dir, 'clip_fo.lancedb'),
    )
    viz_results = fob.compute_visualization(
        frames_dataset, 
        embeddings=sim_idx,
        patches_field=detections_field, 
        model=model,
        brain_key="emb_viz",
        batch_size=batch_size, 
        method=proj,
    )
    # sim_idx.find_unique(n_unique)
    # plot = sim_idx.visualize_unique(viz_results)
    # plot.show(height=800, yaxis_scaleanchor="x")
    # unique_dataset = frames_dataset.select(sim_idx.unique_ids)


    patches_dataset = frames_dataset.to_patches(detections_field)
    session = fo.launch_app(patches_dataset)
    embed()
    # input("aaa")
    # input("aaa")
    # input("aaa")

    # if export_dataset_dir:
    #     patches_dataset.export(
    #         export_dir=export_dataset,
    #         dataset_type=fo.types.FiftyOneVideoLabelsDataset,
    #         label_field=detections_field,
    #         export_media="symlink",
    #     )


if __name__ == '__main__':
    import fire
    fire.Fire(main)