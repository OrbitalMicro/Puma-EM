#include <cstdio>
#include <complex>
#include <fstream>

#include <blitz/array.h>
#include <mpi.h>

#include "rhs_cache.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const char* filename = "test_rhs_cache_complex64.bin";
  if (rank == 0) {
    const std::complex<float> values[3] = {
        std::complex<float>(1.0f, 2.0f),
        std::complex<float>(-3.0f, 4.0f),
        std::complex<float>(5.0f, -6.0f)};
    std::ofstream output(filename, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values), sizeof(values));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  blitz::Array<int, 1> local_indices(2);
  local_indices = 2, 0;
  blitz::Array<std::complex<float>, 1> local_rhs;
  loadGlobalRHSCache(local_rhs, filename, 3, local_indices);
  const bool passed =
      local_rhs.size() == 2
      && local_rhs(0) == std::complex<float>(5.0f, -6.0f)
      && local_rhs(1) == std::complex<float>(1.0f, 2.0f);
  int local_failure = passed ? 0 : 1;
  int global_failure = 0;
  MPI_Allreduce(
      &local_failure, &global_failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::remove(filename);
  MPI_Finalize();
  return global_failure;
}