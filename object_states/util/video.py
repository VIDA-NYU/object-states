import os
import cv2
import numpy as np
import supervision as sv



# ---------------------------------------------------------------------------- #
#                                    Drawing                                   #
# ---------------------------------------------------------------------------- #

class DetectionAnnotator:
    def __init__(self):
        self.box_ann = sv.BoxAnnotator(text_scale=0.4, text_padding=1)
        self.mask_ann = sv.MaskAnnotator()

    def annotate(self, frame, detections, labels=None):
        if detections is not None:
            frame = self.mask_ann.annotate(frame, detections)
            frame = self.box_ann.annotate(frame, detections, labels=labels)
        return frame


# ---------------------------------------------------------------------------- #
#                                 Video Writers                                #
# ---------------------------------------------------------------------------- #

class DetectionSink(sv.VideoSink):
    def __init__(self, target_path, video_info):
        super().__init__(target_path, video_info)
        self.ann = DetectionAnnotator()

    def write_frame(self, frame, detections=None, labels=None):
        frame = self.ann.annotate(frame, detections, labels)
        return super().write_frame(frame)


class XMemSink(DetectionSink):
    '''Create both a full video
    '''
    def __init__(self, target_path, video_info):
        super().__init__(target_path, video_info)
        self.ann = DetectionAnnotator()
        self.tracks = TrackSink(TrackSink.path_to_track_pattern(target_path), video_info)

    def __enter__(self):
        self.tracks.__enter__()
        return self.__enter__()

    def __exit__(self, *a):
        self.tracks.__exit__(*a)
        return self.__exit__(*a)

    def write_frame(self, frame, detections=None, labels=None):
        self.tracks.write_frame(frame, detections)
        return super().write_frame(frame)


class TrackSink:
    '''Create videos cropped to each track bounding box.
    '''
    def __init__(self, out_format, video_info, size=200, padding=0):
        self.out_format = out_format
        
        os.makedirs(os.path.dirname(self.out_format) or '.', exist_ok=True)
        self.video_info = sv.VideoInfo(width=size, height=size, fps=video_info.fps)
        self.size = (self.video_info.height, self.video_info.width)
        self.padding = padding
        self.writers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for w in self.writers.values():
            w.__exit__(*a)

    def write_frame(self, frame, detections):
        for tid, bbox in zip(detections.tracker_id, detections.xyxy):
            if tid not in self.writers:
                self.writers[tid] = sv.VideoSink(self.out_format.format(tid), self.video_info)
                self.writers[tid].__enter__()
            self._write_frame(self.writers[tid], frame, bbox)

    def _write_frame(self, writer, frame=None, bbox=None):
        if frame is None:
            frame = np.zeros(self.size, dtype='uint8')
        elif bbox is not None:
            x, y, x2, y2 = map(int, bbox)
            frame = frame[y - self.padding:y2 + self.padding, x - self.padding:x2 + self.padding]
        frame = resize_with_pad(frame, self.size)
        writer.write_frame(frame)

    @classmethod
    def path_to_track_pattern(self, path):
        return '{}_track{{}}{}'.format(*os.path.splitext(path))


def resize_with_pad(image, new_shape):
    """Maintains aspect ratio and resizes with padding."""
    original_shape = (image.shape[1], image.shape[0])
    if not all(original_shape):
        return np.zeros(new_shape, dtype=np.uint8)
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)


# ---------------------------------------------------------------------------- #
#                                 Loading Video                                #
# ---------------------------------------------------------------------------- #


def iter_video(video_sample):
    video_path = video_sample.filepath
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)

    for i, frame in tqdm.tqdm(
        enumerate(sv.get_video_frames_generator(video_path), 1),
        total=video_info.total_frames,
        desc=video_path
    ):
        yield i, frame, video_sample.frames[i]


def get_video_info(src, size, fps_down=1, nrows=1, ncols=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
    return video_info, WH



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
