#ifndef PUMA_WARM_START_H
#define PUMA_WARM_START_H

#include <complex>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

enum WarmStartStatus {
  WARM_START_ABSENT = 0,
  WARM_START_LOADED = 1,
  WARM_START_INVALID = 2
};

inline WarmStartStatus readWarmStartComplex64(
    const std::string& filename,
    const int expected_count,
    std::vector<std::complex<float> >& coefficients,
    std::string& error)
{
  coefficients.clear();
  error.clear();
  std::ifstream input(filename.c_str(), std::ios::binary | std::ios::ate);
  if (!input.is_open()) return WARM_START_ABSENT;
  if (expected_count < 0) {
    error = "negative expected coefficient count";
    return WARM_START_INVALID;
  }
  const std::streamoff expected_bytes =
      static_cast<std::streamoff>(expected_count) *
      static_cast<std::streamoff>(sizeof(std::complex<float>));
  const std::streamoff actual_bytes = input.tellg();
  if (actual_bytes != expected_bytes) {
    std::ostringstream message;
    message << "expected " << expected_bytes << " bytes for " << expected_count
            << " complex64 coefficients, found " << actual_bytes;
    error = message.str();
    return WARM_START_INVALID;
  }
  coefficients.resize(expected_count);
  input.seekg(0, std::ios::beg);
  if (expected_bytes > 0) {
    input.read(
        reinterpret_cast<char*>(&coefficients[0]),
        static_cast<std::streamsize>(expected_bytes));
  }
  if (!input) {
    error = "failed while reading warm-start coefficients";
    coefficients.clear();
    return WARM_START_INVALID;
  }
  return WARM_START_LOADED;
}

inline bool selectLocalWarmStart(
    const std::vector<std::complex<float> >& global_coefficients,
    const std::vector<int>& local_rwg_numbers,
    std::vector<std::complex<float> >& local_coefficients,
    std::string& error)
{
  local_coefficients.resize(local_rwg_numbers.size());
  for (std::size_t index = 0; index < local_rwg_numbers.size(); ++index) {
    const int global_index = local_rwg_numbers[index];
    if (global_index < 0 ||
        static_cast<std::size_t>(global_index) >= global_coefficients.size()) {
      std::ostringstream message;
      message << "local RWG index " << global_index
              << " is outside global coefficient vector of size "
              << global_coefficients.size();
      error = message.str();
      local_coefficients.clear();
      return false;
    }
    local_coefficients[index] = global_coefficients[global_index];
  }
  error.clear();
  return true;
}

#endif