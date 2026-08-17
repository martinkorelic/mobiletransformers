// #12: parse_vmrss_kb — labelled "unit-testable without /proc" and never tested until now.

#include <gtest/gtest.h>

#include "mem_probe.h"

TEST(ParseVmRss, ReadsTheValueFromProcStatusShapedText) {
    const std::string status =
            "Name:\tapp\nVmPeak:\t 123456 kB\nVmSize:\t 123400 kB\nVmRSS:\t   45678 kB\nThreads:\t8\n";
    EXPECT_EQ(memprobe::parse_vmrss_kb(status), 45678);
}

TEST(ParseVmRss, HandlesNoWhitespaceAfterTheKey) {
    EXPECT_EQ(memprobe::parse_vmrss_kb("VmRSS:42 kB\n"), 42);
}

TEST(ParseVmRss, ReturnsNegativeOneWhenAbsentOrMalformed) {
    EXPECT_EQ(memprobe::parse_vmrss_kb(""), -1);
    EXPECT_EQ(memprobe::parse_vmrss_kb("VmSize:\t 100 kB\n"), -1);
    EXPECT_EQ(memprobe::parse_vmrss_kb("VmRSS:\t kB\n"), -1);
}
