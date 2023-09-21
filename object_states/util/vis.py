import numpy as np



class TrackVisualizer:
    def __init__(self, video_info=None):
        self.video_info = video_info
        self.fps = getattr(video_info, 'fps', 30)
        self.last_frame = 0
        self.presence = {}

    def update_detections(self, i, track_ids):
        self.last_frame = i
        for tid in track_ids: #detections.tracker_id:
            if tid is not None:
                self.presence[tid].append(i)

    def plot(self, out_dir):
        self.plot_presence(out_dir)

    def plot_presence(self, out_dir):
        track_ids = sorted(self.tracks)
        presence = np.zeros((len(self.tracks), self.last_frame))
        for i, idxs in track_ids:
            presence[i, idxs] = 1

        plt.imshow(track_presence, interpolation='nearest')
        xtick = np.linspace(0, presence.shape[1]-1, 10).astype(int)
        plt.xticks(np.arange(self.last_frame)[xtick] / self.fps / 60)
        plt.yticks(track_ids)
        plt.savefig(f'{out_dir}/tracks.png')

    @classmethod
    def from_sample(cls, sample, field):
        self = cls()
        for i in sample.frames:
            # detections = fo_to_sv(sample.frames[i][field])
            track_ids = [d.index for d in sample.frames[i][field]]
            self.update_detections(i, track_ids)
        return self