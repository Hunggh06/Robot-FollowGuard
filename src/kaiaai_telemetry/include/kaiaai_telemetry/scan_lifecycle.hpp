#ifndef KAAIAI_TELEMETRY__SCAN_LIFECYCLE_HPP_
#define KAAIAI_TELEMETRY__SCAN_LIFECYCLE_HPP_

namespace kaiaai_telemetry
{

template<typename Stamp>
class ScanLifecycle
{
public:
  bool has_scan_start_stamp() const
  {
    return has_scan_start_stamp_;
  }

  const Stamp & scan_start_stamp() const
  {
    return scan_start_stamp_;
  }

  bool complete_scan(const Stamp & stamp)
  {
    if (!has_scan_start_stamp_) {
      scan_start_stamp_ = stamp;
      has_scan_start_stamp_ = true;
    }

    const bool was_broken = broken_scan_ || sequence_gap_pending_;
    sequence_gap_pending_ = false;
    broken_scan_ = false;
    return was_broken;
  }

  void start_next_scan(const Stamp & stamp)
  {
    scan_start_stamp_ = stamp;
    has_scan_start_stamp_ = true;
  }

  void mark_broken()
  {
    broken_scan_ = true;
  }

  void defer_sequence_gap()
  {
    sequence_gap_pending_ = true;
  }

  void finish_message()
  {
    if (sequence_gap_pending_) {
      broken_scan_ = true;
      sequence_gap_pending_ = false;
    }
  }

  bool current_scan_broken() const
  {
    return broken_scan_;
  }

private:
  Stamp scan_start_stamp_{};
  bool has_scan_start_stamp_ = false;
  bool broken_scan_ = false;
  bool sequence_gap_pending_ = false;
};

}

#endif
