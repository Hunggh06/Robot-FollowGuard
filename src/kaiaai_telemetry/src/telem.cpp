// Copyright 2024 REMAKE.AI, KAIA.AI, MAKERSPET.COM
// Ported from kaiaai/kaiaai_telemetry (branch iron) for ROS 2 Humble.
//
// Node nhan raw LDS bytes tren /telemetry, decode bang chinh thu vien LDS
// ma ESP32 dung, roi publish sensor_msgs/LaserScan tren /scan.
//
// Diem khac biet so voi ban goc:
//   - Chi giu LDS02RR (va Neato XV11 no la base class), bo 13 header khac
//   - Bo odom / joint_states / battery / wifi, chi giu /scan
//   - Khong can tep. yaml: tham so co san trong code, doi duoc bang -p
//   - Them tham so use_host_time_stamp (mac dinh false) — xem ghi chu ben
//     _on_scan().
//
// Cac file lds*.h lay nguyen ban tu kaiaai/kaiaai_telemetry/src (Apache-2.0).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "kaiaai_msgs/msg/kaiaai_telemetry2.hpp"
#include "kaiaai_telemetry/diagnostics.hpp"
#include "kaiaai_telemetry/scan_lifecycle.hpp"
#include "kaiaai_telemetry/sequence_tracker.hpp"

#include "lds_lds02rr.h"

namespace {

constexpr double DEG_TO_RAD = 0.017453292519943295;

class KaiaaiTelemetry : public rclcpp::Node
{
public:
  KaiaaiTelemetry()
  : Node("kaiaai_telemetry_node")
  {
    lidar_model_ = declare_parameter<std::string>("laser_scan.lidar_model", "XIAOMI-LDS02RR");
    frame_id_ = declare_parameter<std::string>("laser_scan.frame_id", "laser");
    pub_scan_size_ = declare_parameter<int>("lidar.pub_scan_size", 360);
    angle_offset_deg_ = declare_parameter<double>("lidar.angle_offset_deg", 0.0);
    clockwise_ = declare_parameter<bool>("lidar.clockwise", true);
    range_min_m_ = declare_parameter<double>("lidar.range_min_meters", 0.15);
    range_max_m_ = declare_parameter<double>("lidar.range_max_meters", 6.0);
    mask_radius_m_ = declare_parameter<double>("laser_scan.mask_radius_meters", 0.0);
    min_quality_ = declare_parameter<double>("laser_scan.min_quality", 0.0);
    min_scan_time_ = declare_parameter<double>("laser_scan.min_scan_time", 0.25);
    discard_broken_ = declare_parameter<bool>("laser_scan.discard_broken_scans", false);
    telem_topic_ = declare_parameter<std::string>("telemetry.topic_name_sub", "telemetry");
    scan_topic_ = declare_parameter<std::string>("laser_scan.topic_name_pub", "scan");

    // Tham so tu them, KHONG co trong ban goc.
    //
    // Ban goc lay header.stamp tu chinh telem_msg.stamp, tuc la do ESP32 cap.
    // Firmware Kaia lay do bang rmw_uros_sync_session() + settimeofday() nen
    // dung. Neu ESP32 cua ban khong co buoc do dong bo do, moc se lech
    // 1.79e9 giay va tf2 / slam_toolbox se bo moi scan. Bat tham so nay de
    // node dung gio cua host thay the.
    use_host_time_ = declare_parameter<bool>("use_host_time_stamp", false);

    diag_interval_sec_ = declare_parameter<double>("diagnostics.log_interval_sec", 10.0);
    if (diag_interval_sec_ <= 0.0) {
      diag_interval_sec_ = 10.0;
    }

    if (lidar_model_ != LDS_LDS02RR::get_model_name()) {
      RCLCPP_ERROR(get_logger(),
        "ban port chi ho tro %s, nhan duoc '%s'",
        LDS_LDS02RR::get_model_name().c_str(), lidar_model_.c_str());
      rclcpp::shutdown();
      return;
    }
    if (pub_scan_size_ <= 0) {
      RCLCPP_ERROR(get_logger(), "pub_scan_size phai > 0, nhan %d", pub_scan_size_);
      rclcpp::shutdown();
      return;
    }

    plds_ = new LDS_LDS02RR();
    plds_->setReadByteCallback(&KaiaaiTelemetry::read_byte_cb);
    plds_->setScanPointCallback(&KaiaaiTelemetry::scan_point_cb);

    // INFINITY chứ không phải 0.0f. Tia không đo được phải là "no return",
    // mà LaserScan quy ước "no return" = inf. Giá trị 0 nghĩa là vật cản nằm
    // đúng tại sensor: RViz vẽ vệt tỏa ra tâm, slam_toolbox thấy vật cản
    // dính vào robot. Bản gốc Kaia điền 0.0f — giữ nguyên sẽ tái diễn đúng
    // lỗi đó.
    ranges_.assign(pub_scan_size_, INFINITY);
    intensities_.assign(pub_scan_size_, 0.0f);

    telem_sub_ = create_subscription<kaiaai_msgs::msg::KaiaaiTelemetry2>(
      telem_topic_, rclcpp::SensorDataQoS(),
      [this](const kaiaai_msgs::msg::KaiaaiTelemetry2 & msg) { this->_on_scan(msg); });

    scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(scan_topic_, 10);

    diag_timer_ = create_wall_timer(
      std::chrono::duration<double>(diag_interval_sec_),
      [this]() { this->_emit_diagnostics(); });

    RCLCPP_INFO(get_logger(),
      "LDS %s -> %s (%d diem, frame %s), stamp=%s",
      lidar_model_.c_str(), scan_topic_.c_str(), pub_scan_size_, frame_id_.c_str(),
      use_host_time_ ? "host" : "ESP32");
  }

  ~KaiaaiTelemetry() override
  {
    delete plds_;
    plds_ = nullptr;
  }

private:
  void _emit_diagnostics()
  {
    RCLCPP_INFO(get_logger(),
      "[DIAG] seq_gap=%u msgs=%u crc=%u invalid=%u scans_ok=%u scans_drop=%u valid_points=%u",
      diag_.sequence_gaps, diag_.messages, diag_.crc_errors,
      diag_.invalid_packets, diag_.scans_ok, diag_.scans_dropped,
      diag_.valid_points);

    diag_.reset();
  }

  static int read_byte_cb(const void * context)
  {
    void * ctx = const_cast<void *>(context);
    return reinterpret_cast<KaiaaiTelemetry*>(ctx)->_read_byte();
  }

  static void scan_point_cb(const void * context,
    float angle_deg, float distance_mm, float quality, float intensity, bool scan_completed)
  {
    void * ctx = const_cast<void *>(context);
    reinterpret_cast<KaiaaiTelemetry*>(ctx)
      ->_on_point(angle_deg, distance_mm, quality, intensity, scan_completed);
  }

  int _read_byte()
  {
    if (pmsg_ == nullptr || idx_ >= pmsg_->lds.size()) {
      return -1;
    }
    return pmsg_->lds[idx_++];
  }

  void _on_scan(const kaiaai_msgs::msg::KaiaaiTelemetry2 & msg)
  {
    const std::uint32_t missing_packets = sequence_tracker_.observe(msg.seq);
    if (missing_packets > 0) {
      scan_lifecycle_.defer_sequence_gap();
      diag_.add_sequence_gap(static_cast<long>(missing_packets) + 1L);
      RCLCPP_INFO(get_logger(), "%u message(s) lost", missing_packets);
    }

    if (msg.lds.empty() || plds_ == nullptr) {
      scan_lifecycle_.finish_message();
      return;
    }

    pmsg_ = &msg;
    idx_ = 0;
    ++msgs_;
    ++diag_.messages;

    while (idx_ < msg.lds.size()) {
      const int err = plds_->decode_data(this);
      if (err == LDS::RESULT_CHECKSUM_ERROR) {
        scan_lifecycle_.mark_broken();
        ++crc_err_;
        ++diag_.crc_errors;
      } else if (err == LDS::RESULT_INVALID_PACKET) {
        scan_lifecycle_.mark_broken();
        ++invalid_pkt_;
        ++diag_.invalid_packets;
      } else if (err != LDS::RESULT_OK && err != LDS::RESULT_NOT_READY) {
        RCLCPP_WARN(get_logger(), "decode_data tra ve %d", err);
      }
    }
    pmsg_ = nullptr;
    scan_lifecycle_.finish_message();
  }

  void _on_point(float angle_deg, float distance_mm, float quality, float intensity,
    bool scan_completed)
  {
    (void)quality;
    (void)intensity;

    if (scan_completed) {
      _publish();
      std::fill(ranges_.begin(), ranges_.end(), INFINITY);
      std::fill(intensities_.begin(), intensities_.end(), 0.0f);
      valid_pts_ = 0;
      crc_err_ = 0;
      invalid_pkt_ = 0;
      msgs_ = 0;
    }

    if (distance_mm == 0.0f) {        // tia khong do duoc
      return;
    }

    if (min_quality_ > 0.0 && quality < min_quality_) {  // tin hieu yeu: nhieu
      return;
    }

    const double meters = distance_mm * 0.001;
    if (mask_radius_m_ >= meters) {   // qua gan, bo
      return;
    }

    double angle = clockwise_ ? angle_deg : 360.0 - angle_deg;
    angle = std::fmod(angle, 360.0);
    if (angle < 0) {
      angle += 360.0;
    }

    const double increment_deg = 360.0 / pub_scan_size_;
    const int idx = static_cast<int>(std::lround(angle / increment_deg));

    if (idx >= 0 && idx < static_cast<int>(ranges_.size())) {
      ranges_[idx] = static_cast<float>(meters);
      intensities_[idx] = quality;
      ++valid_pts_;
      ++diag_.valid_points;
    }
  }

  void _publish()
  {
    const bool was_broken = scan_lifecycle_.complete_scan(pmsg_->stamp);

    if (was_broken && discard_broken_) {
      ++scans_drop_;
      ++diag_.scans_dropped;
      scan_lifecycle_.start_next_scan(pmsg_->stamp);
      return;
    }

    sensor_msgs::msg::LaserScan out;
    out.header.frame_id = frame_id_;
    now_ = now();
    out.header.stamp = use_host_time_ ? now_ : scan_lifecycle_.scan_start_stamp();
    out.angle_min = angle_offset_deg_ * DEG_TO_RAD;
    out.angle_increment = (360.0 / pub_scan_size_) * DEG_TO_RAD;
    out.angle_max = (angle_offset_deg_ +
      360.0 * (pub_scan_size_ - 1) / pub_scan_size_) * DEG_TO_RAD;
    out.range_min = static_cast<float>(range_min_m_);
    out.range_max = static_cast<float>(range_max_m_);
    out.ranges = ranges_;
    out.intensities = intensities_;

    // scan_time tu chieu lech giua hai lan publish
    const float dt = static_cast<float>(
      pmsg_->stamp.sec - scan_lifecycle_.scan_start_stamp().sec +
      (pmsg_->stamp.nanosec - scan_lifecycle_.scan_start_stamp().nanosec) * 1e-9);

    if (!was_broken && dt > 0.0f && dt <= static_cast<float>(min_scan_time_)) {
      last_valid_scan_time_ = dt;
    }
    out.scan_time = last_valid_scan_time_;
    out.time_increment = out.scan_time / std::max(1, pub_scan_size_ - 1);

    ++scans_ok_;
    ++diag_.scans_ok;

    scan_lifecycle_.start_next_scan(pmsg_->stamp);
    scan_pub_->publish(out);
  }

  std::string lidar_model_;
  std::string frame_id_;
  std::string telem_topic_;
  std::string scan_topic_;
  int pub_scan_size_;
  double angle_offset_deg_;
  bool clockwise_;
  double range_min_m_;
  double range_max_m_;
  double mask_radius_m_;
  double min_quality_;
  double min_scan_time_;
  bool discard_broken_;
  bool use_host_time_;
  double diag_interval_sec_;

  rclcpp::Subscription<kaiaai_msgs::msg::KaiaaiTelemetry2>::SharedPtr telem_sub_;
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
  rclcpp::TimerBase::SharedPtr diag_timer_;

  LDS * plds_ = nullptr;
  const kaiaai_msgs::msg::KaiaaiTelemetry2 * pmsg_ = nullptr;
  size_t idx_ = 0;
  unsigned valid_pts_ = 0;
  unsigned crc_err_ = 0;
  unsigned invalid_pkt_ = 0;
  unsigned msgs_ = 0;
  unsigned scans_ok_ = 0;
  unsigned scans_drop_ = 0;
  kaiaai_telemetry::DiagnosticCounters diag_;
  kaiaai_telemetry::ScanLifecycle<builtin_interfaces::msg::Time> scan_lifecycle_;
  kaiaai_telemetry::SequenceTracker sequence_tracker_;
  std::vector<float> ranges_;
  std::vector<float> intensities_;
  builtin_interfaces::msg::Time now_;
  float last_valid_scan_time_ = 0.0f;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KaiaaiTelemetry>());
  rclcpp::shutdown();
  return 0;
}
