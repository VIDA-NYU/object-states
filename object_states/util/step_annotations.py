import os
import tqdm
import pathtrees as pt
import pandas as pd
import fiftyone as fo



pathn = pt.Path('{key}_{skill}_{medium}_{part_id:d}_video-{vid_id:04d}_{cam}.mp4')

def fname_to_video_id(fname):
    # translate filename to recipe ID
    video_id = os.path.basename(fname)
    try:
        pdata = pathn.parse(video_id)
        pdata['rid'] = ord(pdata['key'].upper()) - 65
    except ValueError:
        import traceback
        traceback.print_exc()
        return
    video_id = 'R{rid}-P{part_id:02d}_{vid_id:02d}'.format(**pdata)
    return video_id

def add_step_annotations(view, steps_csv):
    steps_df = pd.read_csv(steps_csv)

    for d in tqdm.tqdm(view):
        video_id = fname_to_video_id(d.filepath)
        if not video_id: 
            continue

        # add steps
        sdf = steps_df[steps_df.video_id == video_id]
        d["steps"] = fo.TemporalDetections(
            detections=[
                fo.TemporalDetection(label=row.narration, support=[int(row.start_frame)+1, int(row.stop_frame)+1])
                for _, row in sdf.iterrows()
            ]
        )
        d.save()
    return view

