#ifndef KAAIAI_TELEMETRY__DIAGNOSTICS_HPP_
#define KAAIAI_TELEMETRY__DIAGNOSTICS_HPP_

#include <cstdint>

namespace kaiaai_telemetry
{

struct DiagnosticCounters
{
  std::uint32_t sequence_gaps = 0;
  std::uint32_t messages = 0;
  std::uint32_t crc_errors = 0;
  std::uint32_t invalid_packets = 0;
  std::uint32_t scans_ok = 0;
  std::uint32_t scans_dropped = 0;
  std::uint32_t valid_points = 0;

  void add_sequence_gap(long sequence_difference)
  {
    if (sequence_difference > 1) {
      sequence_gaps += static_cast<std::uint32_t>(sequence_difference - 1);
    }
  }

  void reset()
  {
    sequence_gaps = 0;
    messages = 0;
    crc_errors = 0;
    invalid_packets = 0;
    scans_ok = 0;
    scans_dropped = 0;
    valid_points = 0;
  }
};

}

#endif
