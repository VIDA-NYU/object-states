import os
import glob
import tqdm
from collections import Counter
import cv2
import numpy as np
import supervision as sv



# ---------------------------------------------------------------------------- #
#                                    Drawing                                   #
# ---------------------------------------------------------------------------- #

class DetectionAnnotator:
    def __init__(self, **kw):
        self.box_ann = sv.BoxAnnotator(text_scale=0.4, text_padding=1, **kw)
        self.mask_ann = sv.MaskAnnotator(**kw)

    def annotate(self, frame, detections, labels=None, by_track=False):
        if by_track:
            detections.class_id = detections.tracker_id
        frame = self.mask_ann.annotate(frame, detections)
        frame = self.box_ann.annotate(frame, detections, labels=labels, skip_label=labels is None)
        return frame


# ---------------------------------------------------------------------------- #
#                                 Video Writers                                #
# ---------------------------------------------------------------------------- #

class VideoSink(sv.VideoSink):
    def write_frame(self, frame):
        sh = frame.shape[:2]
        she = (self.video_info.height, self.video_info.width)
        assert sh == she, f"Frame must be size: {sh}. Got {she}"
        return super().write_frame(frame)


class DetectionSink(VideoSink):
    def __init__(self, target_path, video_info):
        super().__init__(target_path, video_info)
        self.ann = DetectionAnnotator()

    def write_frame(self, frame, detections=None, labels=None):
        frame = self.ann.annotate(frame, detections, labels)
        return super().write_frame(frame)


class XMemSink(VideoSink):
    '''Create both a full video
    '''
    def __init__(self, target_path, video_info, **kw):
        os.makedirs(target_path, exist_ok=True)
        super().__init__(f'{target_path}/full.mp4', video_info)
        # self.ann = DetectionAnnotator()
        self.tracks = TrackSink(target_path, video_info, **kw)

    def __enter__(self):
        self.tracks.__enter__()
        return super().__enter__()

    def __exit__(self, *a):
        self.tracks.__exit__(*a)
        return super().__exit__(*a)

    # def write_frame(self, frame, detections=None, labels=None):
    #     self.tracks.write_frame(frame.copy(), detections)
    #     frame = self.ann.annotate(frame.copy(), detections, labels)
    #     return super().write_frame(frame)


class TrackSink:
    '''Create videos cropped to each track bounding box.
    '''
    def __init__(self, out_dir, video_info, size=200, padding=0, min_frames=10, remove_existing=True):
        self.out_dir = out_dir
        self.video_info = sv.VideoInfo(width=size, height=size, fps=video_info.fps)
        self.size = (self.video_info.height, self.video_info.width)
        self.padding = padding
        self.min_frames = min_frames
        self.writers = {}
        if remove_existing:
            self.remove_track_videos()

    def remove_track_videos(self):
        for f in glob.glob(f'{self.out_dir}track_*.mp4'):
            print(f"Deleting existing track video {f}")
            os.remove(f)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for w in self.writers.values():
            w.__exit__(*a)
            if w.count < self.min_frames:
                os.remove(w.target_path)
            else:
                f = w.target_path
                label = w.label_counts.most_common()
                label = (label or [[None]])[0][0]
                f2 = '{}_{l}_{s}-{e}{}'.format(
                    *os.path.splitext(f), 
                    s=w.first_seen, 
                    e=w.last_seen,
                    l=label or 'noclass')
                os.rename(f, f2)

    def write_frame(self, frame, detections, labels=None, frame_idx=None):
        if labels is None:
            labels = [None]*len(detections)
        for tid, bbox, label in zip(detections.tracker_id, detections.xyxy, labels):
            self._write_frame(self._get_writer(tid, frame_idx), frame, bbox, label, frame_idx)

    def _get_writer(self, tid, frame_idx):
        if tid in self.writers:
            return self.writers[tid]
        fname = f'{self.out_dir}/track_{tid}.mp4' #self.out_format.format(tid)
        os.makedirs(os.path.dirname(fname) or '.', exist_ok=True)
        self.writers[tid] = w = VideoSink(fname, self.video_info)
        w.__enter__()
        w.count = 0
        w.track_id = tid
        w.label_counts = Counter()
        w.first_seen = frame_idx or -1
        w.last_seen = frame_idx or -1
        return w

    def _write_frame(self, writer, frame=None, bbox=None, label=None, frame_idx=None):
        if frame is None:
            frame = np.zeros(self.size, dtype='uint8')
        elif bbox is not None:
            frame = crop_box(frame, bbox, self.padding)
        frame = resize_with_pad(frame, self.size)
        writer.write_frame(frame)
        writer.count += 1
        if frame_idx is not None:
            writer.last_seen = frame_idx
        if label is not None:
            writer.label_counts.update([label])


def crop_box(frame, bbox, padding=0):
    H, W = frame.shape[:2]
    x, y, x2, y2 = map(int, bbox)
    return frame[
        max(y - padding, 0):min(y2 + padding, H), 
        max(x - padding, 0):min(x2 + padding, W)]



def crop_box_with_size(frame, bbox, target_size, padding=0):
    H, W = frame.shape[:2]
    x, y, x2, y2 = map(int, bbox)
    
    # Calculate the current width and height of the bounding box
    current_width = x2 - x
    current_height = y2 - y
    current_aspect_ratio = current_width / current_height
    
    # Calculate the center of the bounding box
    x_center = (x + x2) / 2
    y_center = (y + y2) / 2
    
    # Calculate the new bounding box dimensions to match the desired size
    final_width, final_height = target_size
    target_aspect_ratio = final_width / final_height

    # If an aspect_ratio is specified, adjust the bounding box to match it
    new_width, new_height = (
        (current_width, int(current_width / target_aspect_ratio))
        if current_aspect_ratio > target_aspect_ratio else 
        (int(current_height * target_aspect_ratio), current_height)
    )
    # print(current_width, current_height)
    # print(new_width, new_height)
    # print(final_width, final_height)

    # Calculate the new coordinates for the bounding box
    x = max(int(x_center - new_width / 2), 0)
    y = max(int(y_center - new_height / 2), 0)
    x2 = min(int(x_center + new_width / 2), W)
    y2 = min(int(y_center + new_height / 2), H)

    # Crop the region
    cropped_region = frame[
        max(y - padding, 0):min(y2 + padding, H),
        max(x - padding, 0):min(x2 + padding, W)]
    
    # Resize the cropped region to the desired size
    resized_cropped_region = cv2.resize(cropped_region, target_size)

    return resized_cropped_region


def resize_with_pad(image, new_shape):
    """Maintains aspect ratio and resizes with padding."""
    original_shape = (image.shape[1], image.shape[0])
    if not all(original_shape):
        return np.zeros(new_shape, dtype=np.uint8)
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([max(int(x*ratio), 1) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------- #
#                                 Loading Video                                #
# ---------------------------------------------------------------------------- #


def iter_video(video_sample, pbar=False):
    video_path = video_sample.filepath
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)

    it = tqdm.tqdm(
        enumerate(sv.get_video_frames_generator(video_path), 1),
        total=video_info.total_frames,
        desc=video_path
    )
    for i, frame in it:
        finfo = video_sample.frames[i]
        yield (i, frame, finfo, it) if pbar else (i, frame, finfo)


# def get_video_info(src, size, fps_down=1, nrows=1, ncols=1):
#     # get the source video info
#     video_info = sv.VideoInfo.from_video_path(video_path=src)
#     # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
#     aspect = video_info.width / video_info.height
#     video_info.width = int(aspect*size)//16*16
#     video_info.height = int(size)//16*16
#     WH = video_info.width, video_info.height

#     # double width because we have both detic and xmem frames
#     video_info.width *= ncols
#     video_info.height *= nrows
#     # possibly reduce the video frame rate
#     video_info.og_fps = video_info.fps
#     video_info.fps /= fps_down or 1

#     print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
#     return video_info, WH


def get_video_info(src, size, fps_down=1, nrows=1, ncols=1, render_scale=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    print(f'Input Video {src}')
    print(f"Original size: {[video_info.width, video_info.height]}")
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height
    video_info.width *= render_scale
    video_info.height *= render_scale
    WH2 = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"size: {WH} {WH2} grid={ncols}x{nrows}  fps: {video_info.fps} ({fps_down or 1}x)")
    return video_info, WH, WH2


# ---------------------------------------------------------------------------- #
#                                  Conversion                                  #
# ---------------------------------------------------------------------------- #


def detectron2_to_sv(outputs, labels):
    detections = sv.Detections(
        xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
        mask=outputs["instances"].pred_masks.int().cpu().numpy(),
        confidence=outputs["instances"].scores.cpu().numpy(),
        class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
    )
    labels = labels[detections.class_id]
    return detections, labels

def tracks_to_sv(pred_mask, label_counts, track_ids, labels):
    # convert to Detection object for visualization
    track_detections = sv.Detections(
        mask=pred_mask.cpu().numpy(),
        xyxy=masks_to_boxes(pred_mask).cpu().numpy(),
        class_id=np.array([label_counts[i].most_common(1)[0][0] for i in track_ids]),
        tracker_id=track_ids,
    )

    # draw xmem detections
    labels = [
        [f'{i} {labels[l][:12]} {c}' 
            for l, c in label_counts[i].most_common(2)] 
        for i in track_detections.tracker_id
    ]
    return track_detections, labels



def backup_path(path, **kw):
    if os.path.exists(path):
        p2 = next_path(path, **kw)
        print(f"Moving {path} to {p2}")
        os.rename(path, p2)
    return path

def next_path(path, sep='_'):
    path_pattern = '{}{sep}{{}}{}'.format(*os.path.splitext(path), sep=sep)
    i = 1
    while os.path.exists(path_pattern.format(i)):
        i += 1
    return path_pattern.format(i)