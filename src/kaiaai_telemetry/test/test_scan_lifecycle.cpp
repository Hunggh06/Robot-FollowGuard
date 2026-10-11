#include <cassert>
#include <cstdint>

#include "kaiaai_telemetry/scan_lifecycle.hpp"

namespace
{

struct TestStamp
{
  std::int32_t sec = 0;
  std::uint32_t nanosec = 0;
};

}

int main()
{
  kaiaai_telemetry::ScanLifecycle<TestStamp> zero_stamp_lifecycle;
  const TestStamp zero_stamp{};

  assert(!zero_stamp_lifecycle.has_scan_start_stamp());
  assert(!zero_stamp_lifecycle.complete_scan(zero_stamp));
  assert(zero_stamp_lifecycle.has_scan_start_stamp());
  assert(zero_stamp_lifecycle.scan_start_stamp().sec == 0);
  assert(zero_stamp_lifecycle.scan_start_stamp().nanosec == 0);

  kaiaai_telemetry::ScanLifecycle<TestStamp> gap_lifecycle;
  const TestStamp first_stamp{1, 0};
  const TestStamp second_stamp{2, 0};
  const TestStamp third_stamp{3, 0};

  assert(!gap_lifecycle.complete_scan(first_stamp));
  gap_lifecycle.defer_sequence_gap();
  assert(gap_lifecycle.complete_scan(second_stamp));
  assert(!gap_lifecycle.current_scan_broken());
  assert(!gap_lifecycle.complete_scan(third_stamp));

  gap_lifecycle.defer_sequence_gap();
  gap_lifecycle.finish_message();
  assert(gap_lifecycle.current_scan_broken());

  assert(gap_lifecycle.complete_scan(third_stamp));
  gap_lifecycle.start_next_scan(third_stamp);
  assert(!gap_lifecycle.current_scan_broken());
  assert(!gap_lifecycle.complete_scan(first_stamp));
}
