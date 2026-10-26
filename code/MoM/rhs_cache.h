#ifndef PUMA_RHS_CACHE_H
#define PUMA_RHS_CACHE_H

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <string>

#include <blitz/array.h>
#include <mpi.h>

inline std::string readSingleLine(const std::string& filename)
{
  std::ifstream input(filename.c_str());
  if (!input.is_open()) {
    std::cerr << "Unable to open " << filename << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  std::string value;
  getline(input, value);
  return value;
}

inline void loadGlobalRHSCache(
    blitz::Array<std::complex<float>, 1>& local_rhs,
    const std::string& filename,
    const int N_RWG,
    const blitz::Array<int, 1>& localRWGNumbers)
{
  const unsigned int endian_probe = 1;
  const bool little_endian =
      *reinterpret_cast<const unsigned char*>(&endian_probe) == 1;
  if (!little_endian || sizeof(std::complex<float>) != 2 * sizeof(float)) {
    std::cerr << "RHS cache requires little-endian interleaved complex64"
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  std::ifstream input(filename.c_str(), std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    std::cerr << "Unable to open RHS cache " << filename << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  const std::streamoff expected_size =
      static_cast<std::streamoff>(N_RWG) * sizeof(std::complex<float>);
  if (input.tellg() != expected_size) {
    std::cerr << "RHS cache size does not equal 8 * N_RWG" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  input.seekg(0, std::ios::beg);
  blitz::Array<std::complex<float>, 1> global_rhs(N_RWG);
  input.read(reinterpret_cast<char*>(global_rhs.data()), expected_size);
  if (!input || input.gcount() != expected_size) {
    std::cerr << "RHS cache read was incomplete" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  local_rhs.resize(localRWGNumbers.size());
  for (int local_index = 0; local_index < localRWGNumbers.size(); ++local_index) {
    const int global_index = localRWGNumbers(local_index);
    if (global_index < 0 || global_index >= N_RWG) {
      std::cerr << "RHS cache local-to-global RWG index is out of range"
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const std::complex<float> value = global_rhs(global_index);
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      std::cerr << "RHS cache contains a non-finite value" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    local_rhs(local_index) = value;
  }
}

#endif