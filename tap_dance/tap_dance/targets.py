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
    way each gets a tolerance: half the distance to its NEAREST NEIGHBOUR, capped
    by max_halfwidth. A single global halfwidth taken from the tightest pair would
    make every region as strict as the worst one -- an isolated target 450 px from
    anything else would still reject a tap 60 px off. The cap exists only so a
    lone target cannot claim the whole image.
    """

    def __init__(self, classes=(), min_hits=15, smoothing=0.2,
                 max_halfwidth=400.0, scale=1.0):
        self._classes = set(classes)
        self._min_hits = min_hits
        self._smoothing = smoothing
        self._max_hw = max_halfwidth
        self._scale = scale
        self._seen = {}        # name -> [hits, smoothed_u, best_score]
        self.targets = []      # [(name, u)] sorted by u
        self.tolerance = {}    # name -> px

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
        self._recompute(stable, keep_tolerance=True)
        return False

    def _recompute(self, targets, keep_tolerance=False):
        self.targets = sorted(targets, key=lambda t: t[1])
        if keep_tolerance and self.tolerance:
            return
        self.tolerance = {}
        for i, (name, cu) in enumerate(self.targets):
            others = [abs(cu - ou) for j, (_, ou) in enumerate(self.targets) if j != i]
            self.tolerance[name] = min(
                self._max_hw, (min(others) / 2.0) if others else self._max_hw)

    # --- queries ---------------------------------------------------------
    def match(self, u):
        """Nearest target whose own tolerance contains u, or None."""
        best, best_d = None, float('inf')
        for name, cu in self.targets:
            d = abs(u - cu)
            if d <= self.tolerance[name] and d < best_d:
                best, best_d = name, d
        return best

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
        return ',  '.join(f'{n}@{int(u)}+/-{self.tolerance[n]:.0f}'
                          for n, u in self.targets)
