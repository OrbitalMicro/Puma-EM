#include "warm_start.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void writeCoefficients(
    const std::string& path,
    const std::vector<std::complex<float> >& coefficients)
{
  std::ofstream output(path.c_str(), std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(&coefficients[0]),
      coefficients.size() * sizeof(std::complex<float>));
}

}  // namespace

int main()
{
  const std::string valid_path = "test_warm_start_valid.bin";
  const std::string invalid_path = "test_warm_start_invalid.bin";
  std::vector<std::complex<float> > expected;
  expected.push_back(std::complex<float>(1.0f, -1.0f));
  expected.push_back(std::complex<float>(2.0f, -2.0f));
  expected.push_back(std::complex<float>(3.0f, -3.0f));
  expected.push_back(std::complex<float>(4.0f, -4.0f));
  writeCoefficients(valid_path, expected);
  std::vector<std::complex<float> > truncated(expected.begin(), expected.end() - 1);
  writeCoefficients(invalid_path, truncated);

  std::vector<std::complex<float> > loaded;
  std::string error;
  assert(
      readWarmStartComplex64(valid_path, expected.size(), loaded, error) ==
      WARM_START_LOADED);
  assert(loaded == expected);
  assert(error.empty());

  std::vector<int> local_indices;
  local_indices.push_back(3);
  local_indices.push_back(0);
  local_indices.push_back(2);
  std::vector<std::complex<float> > local;
  assert(selectLocalWarmStart(loaded, local_indices, local, error));
  assert(local[0] == expected[3]);
  assert(local[1] == expected[0]);
  assert(local[2] == expected[2]);

  assert(
      readWarmStartComplex64(invalid_path, expected.size(), loaded, error) ==
      WARM_START_INVALID);
  assert(!error.empty());
  assert(
      readWarmStartComplex64("missing_warm_start.bin", expected.size(), loaded, error) ==
      WARM_START_ABSENT);

  local_indices[0] = 4;
  assert(!selectLocalWarmStart(expected, local_indices, local, error));
  assert(!error.empty());

  std::remove(valid_path.c_str());
  std::remove(invalid_path.c_str());
  std::cout << "warm-start tests passed" << std::endl;
  return 0;
}