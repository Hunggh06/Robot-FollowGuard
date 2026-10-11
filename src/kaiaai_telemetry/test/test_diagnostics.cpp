#include <cassert>
#include <cstdint>

#include "kaiaai_telemetry/diagnostics.hpp"
#include "kaiaai_telemetry/sequence_tracker.hpp"

int main()
{
  kaiaai_telemetry::SequenceTracker contiguous_tracker;
  assert(contiguous_tracker.observe(static_cast<std::uint32_t>(100)) == 0);
  assert(contiguous_tracker.observe(static_cast<std::uint32_t>(101)) == 0);
  assert(contiguous_tracker.observe(static_cast<std::uint32_t>(101)) == 0);
  assert(contiguous_tracker.observe(static_cast<std::uint32_t>(100)) == 0);

  kaiaai_telemetry::SequenceTracker jump_tracker;
  assert(jump_tracker.observe(static_cast<std::uint32_t>(100)) == 0);
  assert(jump_tracker.observe(static_cast<std::uint32_t>(104)) == 3);

  kaiaai_telemetry::DiagnosticCounters counters;
  counters.add_sequence_gap(5);
  counters.messages = 2;
  counters.crc_errors = 3;
  counters.invalid_packets = 4;
  counters.scans_ok = 6;
  counters.scans_dropped = 7;
  counters.valid_points = 8;

  assert(counters.sequence_gaps == 4);
  assert(counters.messages == 2);
  assert(counters.crc_errors == 3);
  assert(counters.invalid_packets == 4);
  assert(counters.scans_ok == 6);
  assert(counters.scans_dropped == 7);
  assert(counters.valid_points == 8);

  counters.reset();
  assert(counters.sequence_gaps == 0);
  assert(counters.messages == 0);
  assert(counters.crc_errors == 0);
  assert(counters.invalid_packets == 0);
  assert(counters.scans_ok == 0);
  assert(counters.scans_dropped == 0);
  assert(counters.valid_points == 0);
}
