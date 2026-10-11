#ifndef KAAIAI_TELEMETRY__SEQUENCE_TRACKER_HPP_
#define KAAIAI_TELEMETRY__SEQUENCE_TRACKER_HPP_

#include <cstdint>

namespace kaiaai_telemetry
{

class SequenceTracker
{
public:
  std::uint32_t observe(std::uint32_t sequence)
  {
    if (!initialized_) {
      last_sequence_ = sequence;
      initialized_ = true;
      return 0;
    }

    if (sequence <= last_sequence_) {
      return 0;
    }

    const std::uint32_t missing = sequence - last_sequence_ - 1;
    last_sequence_ = sequence;
    missing_packets_ += missing;
    return missing;
  }

  std::uint32_t missing_packets() const
  {
    return missing_packets_;
  }

private:
  bool initialized_ = false;
  std::uint32_t last_sequence_ = 0;
  std::uint32_t missing_packets_ = 0;
};

}

#endif
