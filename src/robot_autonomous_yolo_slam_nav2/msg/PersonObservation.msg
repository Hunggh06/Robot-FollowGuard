std_msgs/Header header
int32 track_id
float32 confidence
float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height
string state
bool valid
# Optional validated foot-floor contact pixel from pose keypoints.
# False when feet are occluded, cropped, sitting, or keypoints weak.
bool footpoint_valid
float32 footpoint_x
float32 footpoint_y
float32 footpoint_confidence
