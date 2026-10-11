std_msgs/Header header
int32 track_id
float32 confidence
float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height
string state
bool valid
bool identity_confirmed
geometry_msgs/Point point_base_link
float32 range_m
float32 bearing_rad
# Validated foot-floor contact pixel from the SAME pose inference frame.
# The YOLO node sets these; the metric node consumes them for the
# camera-floor fallback. Invalid (false) means no defensible feet evidence.
bool footpoint_valid
float32 footpoint_x
float32 footpoint_y
float32 footpoint_confidence
# Provenance of point_base_link/range_m/bearing_rad. One of the
# POSITION_SOURCE_* constants below. "held_invalid" preserves the last
# valid point for display only with valid=false (never a fresh Nav2 goal).
# "none" means no geometry. header.stamp is the measurement stamp of the
# live source, or the ORIGINAL measurement stamp for held_invalid.
string POSITION_SOURCE_NONE=none
string POSITION_SOURCE_LIDAR=lidar
string POSITION_SOURCE_CAMERA=camera
string POSITION_SOURCE_FUSED=fused
string POSITION_SOURCE_HELD_INVALID=held_invalid
string position_source
# Geometry confidence of the position ONLY (0..1). Independent of the
# person identity confidence in `confidence`; never fabricate identity
# from floor geometry.
float32 position_confidence
# Short machine-readable reason, e.g. "lidar_bbox",
# "held track=3 age=1.2s", "no_geometry".
string provenance
