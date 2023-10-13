import os
import tqdm
import pathtrees as pt
import pandas as pd



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
    import fiftyone as fo
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




def load_object_annotations(cfg):
    meta_df = pd.read_csv(cfg.DATASET.META_CSV).set_index('video_name').groupby(level=0).first()
    object_names = []
    for c in meta_df.columns:
        if c.startswith('#'):
            meta_df[c[1:]] = meta_df.pop(c).fillna('').apply(lambda s: [int(float(x)) for x in str(s).split('+') if x != ''])
            object_names.append(c[1:])

    states_df = pd.read_csv(cfg.DATASET.STATES_CSV)
    states_df = states_df[states_df.time.fillna('') != '']
    states_df['time'] = pd.to_timedelta(states_df.time.apply(lambda x: f'00:{x}'))
    states_df['start_frame'] = (states_df.time.dt.total_seconds() * meta_df.fps.loc[states_df.video_name].values).astype(int)
    
    # creating a dict of {video_id: {track_id: df}}
    dfs = {}
    for vid, row in meta_df.iterrows():
        objs = {}
        sdf = states_df[states_df.video_name == vid]
        for c in object_names:
            if c not in sdf.columns:
                continue
            odf = sdf[[c, 'start_frame']].copy().rename(columns={c: "state"})
            odf = odf[odf.state.fillna('') != '']
            odf['stop_frame'] = odf['start_frame'].shift(-1)
            odf['object'] = c
            if not len(odf):
                continue
            for track_id in row[c]:
                objs[track_id] = odf
        # if not len(objs):
        #     continue
        dfs[vid] = objs
            
    return dfs

def get_obj_anns(df, frame_idx):
    idxs = []
    for i in frame_idx:
        d = df[(df.start_frame <= i) & (pd.isna(df.stop_frame) | (df.stop_frame > i))]
        idxs.append({
            'state': d.state.iloc[0] if len(d) else None,
            'object': d.object.iloc[0] if len(d) else None,
        })
    return pd.DataFrame(idxs)
