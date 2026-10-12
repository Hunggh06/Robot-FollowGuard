# Algorithms and State Flow

## Identity confirmation

YOLO produces person detections and an IoU tracker assigns transient track
IDs. `PersonTagDetector` converts each person crop to grayscale, can upscale
the crop up to the configured factor, and uses OpenCV's `DICT_APRILTAG_36h11`.
`TagPersonIdentity` accepts exactly one target tag inside exactly one person
box. It rejects duplicate/overlapping attribution rather than guessing.

After confirmation, the target is retained while the same track remains
visible and unambiguous. Reconfirmation is required after loss, ambiguity, a
long gap, or a substantial box discontinuity.

## Metric association and following

The metric node projects fresh LiDAR data into the confirmed image box. A
metric target is valid only when sensor age, skew, calibration, and TF gates
hold. Nav2 receives follow goals only from valid metric data. The follower
uses a follow distance, clearance corridor, and goal-update thresholds from
`person_follow.yaml`.

## Recovery

When visual confirmation disappears, the follower first permits an existing
Nav2 follow action to resolve. Afterward it requests bounded `/spin` actions,
up to `search_max_spin_revolutions`. Fresh tracked-person observations pause a
spin for `min(max_s, people × per_person_s)` so tag inspection can occur; the
default is one second per person, capped at five seconds. A new tag lock exits
recovery immediately.

## Zone guard

Four clicked map points form a polygon. The guard tests the metric target
against it, publishes status, and starts its configured speech/confirmation
sequence. A recognized confirmation can activate guide return; it is not an
independent collision-avoidance system.
