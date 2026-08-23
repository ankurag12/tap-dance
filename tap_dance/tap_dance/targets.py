# SPDX-License-Identifier: Apache-2.0
#
# Shared target logic: turning YOLO detections into targets, and deciding which
# target a horizontal pixel position belongs to.
#
# Lives in its own module so the game and the hover_probe debug tool CANNOT
# disagree. A debug tool that computes "which object is the tag over" differently
# from the game is worse than no tool at all -- it would send you hunting for a
# bug in whichever one you trusted less.
#
# The COCO table lives here rather than in a debug script, so runtime code never
# imports from tap_dance.debug -- dependencies point from the debug tools toward
# this module, never the reverse.


# COCO 80-class order, as used by the Ultralytics YOLOv8 export. The decoder
# publishes the INDEX into this list as class_id (a string), so this is the
# lookup that turns "2" into "car".
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def class_name(class_id):
    """Map the decoder's class_id string to a COCO name; pass through if odd."""
    try:
        return COCO_CLASSES[int(class_id)]
    except (ValueError, IndexError):
        return f'<id {class_id}>'


def network_to_image_scale(image_w, image_h, network_w, network_h):
    """
    Factor converting YOLO bbox pixels into original-image pixels.

    YoloV8DecoderNode declares only tensor_name, the two thresholds and
    num_classes -- it is never told the original image size -- so it emits bboxes
    in the NETWORK's pixel space. AprilTag meanwhile reports the tag in full image
    coordinates, so the two are not comparable without this.

    dnn_image_encoder.launch.py defaults to enable_padding=True and
    keep_aspect_ratio=True: the image is scaled uniformly by
    min(net_w/img_w, net_h/img_h) into the TOP-LEFT of the tensor and the
    remainder padded, after which the crop's computed offset is zero and it is a
    no-op. The inverse is therefore a pure scale with no offset.

    For 1280x720 into 640x640 this is 2.0, confirmed empirically: an apple
    reported at u=143 sat under a tag reading u=290.
    """
    return 1.0 / min(network_w / float(image_w), network_h / float(image_h))


class TargetSet:
    """
    The set of tappable targets and the rule for matching a tap to one.

    Targets are either fixed (measured by hand) or discovered from YOLO. Either
    way each owns an INTERVAL of columns, and adjacent intervals meet exactly at
    the midpoint between the two targets, so there is no dead band between them.

    Boundaries are computed PER SIDE. An earlier version gave each target a single
    symmetric tolerance equal to half the distance to its nearest neighbour, which
    left gaps whenever spacing was uneven: with targets at 286, 432 and 854, the
    middle one took its +/-73 from its close left neighbour and so stopped at 505,
    while the right one began at 643 -- a 138 px band belonging to nobody.

    Only the OUTER edges of the outermost targets are bounded, by outer_margin, so
    a tap far off to one side is still rejected rather than silently attributed.
    """

    def __init__(self, classes=(), min_hits=15, smoothing=0.2,
                 outer_margin=400.0, scale=1.0):
        self._classes = set(classes)
        self._min_hits = min_hits
        self._smoothing = smoothing
        self._outer = outer_margin
        self._scale = scale
        self._seen = {}        # name -> [hits, smoothed_u, best_score]
        self.targets = []      # [(name, u)] sorted by u
        self.bounds = {}       # name -> (lo, hi) columns owned by this target

    # --- construction ----------------------------------------------------
    def set_static(self, targets):
        self._recompute(list(targets))

    def update_from_detections(self, detections):
        """
        Fold a Detection2DArray's detections in. Returns True if the target SET
        changed (not merely the positions, which drift continuously and would
        otherwise trigger a rebuild every frame).
        """
        promoted = False
        for det in detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            name = class_name(hyp.class_id)
            if self._classes and name not in self._classes:
                continue
            u = det.bbox.center.position.x * self._scale
            entry = self._seen.get(name)
            if entry is None:
                self._seen[name] = [1, u, hyp.score]
            else:
                entry[0] += 1
                a = self._smoothing
                entry[1] = (1.0 - a) * entry[1] + a * u
                entry[2] = max(entry[2], hyp.score)
                if entry[0] == self._min_hits:
                    promoted = True

        stable = [(n, v[1]) for n, v in self._seen.items() if v[0] >= self._min_hits]
        if promoted or {n for n, _ in self.targets} != {n for n, _ in stable}:
            self._recompute(stable)
            return True
        # Positions still move even when the set does not; keep them current.
        self._recompute(stable, keep_bounds=True)
        return False

    def _recompute(self, targets, keep_bounds=False):
        self.targets = sorted(targets, key=lambda t: t[1])
        if keep_bounds and self.bounds:
            return
        self.bounds = {}
        n = len(self.targets)
        for i, (name, cu) in enumerate(self.targets):
            # Halfway to each neighbour; outer_margin only where there is none.
            lo = ((self.targets[i - 1][1] + cu) / 2.0) if i > 0 else cu - self._outer
            hi = ((self.targets[i + 1][1] + cu) / 2.0) if i < n - 1 else cu + self._outer
            self.bounds[name] = (lo, hi)

    # --- queries ---------------------------------------------------------
    def match(self, u):
        """The target owning column u, or None if u is outside every interval.

        Intervals are disjoint and adjacent ones touch, so at most one matches --
        no nearest-neighbour tie-breaking needed.
        """
        for name, _ in self.targets:
            lo, hi = self.bounds[name]
            if lo <= u <= hi:
                return name
        return None

    def ambiguous(self, u, uncertainty):
        """
        True if the uncertainty could change the answer.

        Better than a fixed uncertainty threshold, which once rejected a tap
        uncertain to 101 px against targets 735 px apart -- where 101 px cannot
        possibly confuse one for the other.
        """
        m = self.match(u)
        return not (self.match(u - uncertainty) == m == self.match(u + uncertainty))

    def hits(self, name):
        entry = self._seen.get(name)
        return entry[0] if entry else 0

    def score(self, name):
        entry = self._seen.get(name)
        return entry[2] if entry else 0.0

    def describe(self):
        return ',  '.join(
            f'{n}@{int(u)}[{int(self.bounds[n][0])}..{int(self.bounds[n][1])}]'
            for n, u in self.targets)
