import os
from IPython import embed
import numpy as np
import pandas as pd
import pathtrees as pt
from .config import get_cfg
pd.options.display.max_rows = 500

def main(config_fname, *video_dirs, key=None, show_objects=False, overwrite=False):
    cfg = get_cfg(config_fname)
    label_files = []
    for ddir in [cfg.DATASET.ROOT, cfg.DATASET.EVAL_ROOT]:
        tree = pt.tree(ddir, {
            'labels': {'{name}.json': 'labels'},
        })

        label_files.extend(tree.labels.glob())

    label_video_names = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
    video_video_names = [os.path.splitext(f)[0] for d in video_dirs or [] for f in os.listdir(d)]

    meta_df = get_sheet(ANN_SHEET, 0, '/datasets/PTG Object State Labels - Metadata.csv', overwrite)
    meta_df = meta_df.dropna(axis=0, subset=['video_name']).groupby('video_name').last().reset_index()
    ignore = meta_df.video_name[meta_df['obj ann method'] != 'no'].dropna().unique()
    meta_fs = meta_df.video_name.dropna().unique()
    bad_tracks = ~pd.isna(meta_df.Notes)
    good_tracks = (~pd.isna(meta_df[[c for c in meta_df.columns if c.startswith('#')]])).any(axis=1)
    meta_df = meta_df.set_index('video_name')

    state_df = get_sheet(ANN_SHEET, 787650247, '/datasets/PTG Object State Labels - State Annotations.csv', overwrite)
    state_df = state_df.dropna(subset=['video_name'])
    state_fs = state_df.video_name.unique()
    ann_state_fs = state_df.video_name[~pd.isna(state_df.time)].dropna().unique()

    all_vids = set(label_video_names) | set(meta_fs) | set(state_fs) | set(video_video_names)

    s = pd.Series(sorted(all_vids))
    df = pd.DataFrame({
        'videos': s.isin(set(video_video_names)),
        'labels': s.isin(set(label_video_names)),
        'tracks': None,
        'ann_states': s.isin(set(ann_state_fs)),
        'meta': s.isin(set(meta_fs)),
    }).set_index(s)
    df.loc[meta_fs, 'tracks'] = pd.Series(list(good_tracks | bad_tracks), index=list(meta_fs))
    df.loc[meta_fs[good_tracks], 'tracks'] = True
    df.loc[meta_fs[bad_tracks & ~good_tracks], 'tracks'] = False
    df = df[df.index.isin(ignore)]
    K={False: 'x', True: '.', pd.isna: '!'}

    print(df.loc[~df.all(1)].sort_values(by=list(df.columns),axis=0).replace(K))
    print(len(df.loc[df.all(1)].replace(K)), 'fully completed out of', len(df))
    print('have')
    print(df.astype(int).sum(0))
    print("missing")
    print(len(df)-df.astype(int).sum(0))

    if key:
        print()
        print('missing', key)
        print('\n'.join(df[key][~df[key]].index.tolist()))

    print()
    # embed()

    if show_objects:
        for c in meta_df.columns:
            o = c.strip('#')
            if c.startswith('#') and o in state_df.columns:
                print(c)

                state_df[f'has_state__{o}'] = ~pd.isna(state_df[o])
                state_df[f'has_track__{o}'] = (~pd.isna(meta_df[c].loc[state_df.video_name])).values
                print(state_df[[o, f'has_track__{o}']].value_counts().unstack().fillna(0).astype(int))
                print()

                x = state_df[state_df[f'has_state__{o}'] & ~state_df[f'has_track__{o}']].video_name.unique()
                if len(x):
                    print(x)

    # print_set("Missing from meta (labels):", set(label_video_names) - set(meta_fs))
    # print_set("Missing from states (labels):", set(label_video_names) - set(state_fs))
    # print_set("Missing from states (meta):", set(meta_fs) - set(state_fs))
    # print_set("Missing from meta (states):", set(state_fs) - set(meta_fs))
    # print_set("Missing from labels (meta):", set(meta_fs) - set(label_video_names))
    # print_set("Missing from labels (states):", set(state_fs) - set(label_video_names))
    print_set("Missing from meta (videos):", set(video_video_names) - set(meta_fs))
    print_set("Missing from states (videos):", set(video_video_names) - set(state_fs))
    # print_set("Unannotated states:", set(video_video_names) - set(ann_state_fs))
    
    
def print_set(msg, xs):
    print(msg, len(xs), ''.join(f'\n  {x}' for x in sorted(xs)))


ANN_SHEET = '13m-DY5QwCHaVrhkkX7EgKJuNTYgXcTaYidFObbmkXKA'
def get_sheet(id, gid=0, cache=None, overwrite=False):
    if cache:
        if not overwrite and os.path.isfile(cache):
            print("Using cache", cache)
            return pd.read_csv(cache)
        print("Querying", cache)
    
    df = pd.read_csv(f'https://docs.google.com/spreadsheets/d/{id}/export?format=csv&gid={gid}')
    df.to_csv(cache)
    return df

if __name__ == '__main__':
    import fire
    fire.Fire(main)