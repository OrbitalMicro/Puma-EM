#include <fstream>
#include <iostream>
#include <string>
#include <complex>
#include <vector>
#include <blitz/array.h>
#include <mpi.h>
#include <map>
#include <set>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <sstream>
#include <stdexcept>

using namespace std;

#include "readWriteBlitzArrayFromFile.h"
#include "GetMemUsage.h"
#include "./lapack/zgels_interface.h"
#include "frob_blocks.h"

class CubeArrays {
  public:
    int number;
    int N_RWG_test;
    int N_RWG_src;
    int N_neighbors;
    std::vector<int> testSrc_RWGsNumbers;
    std::vector<int> isEdgeInCartesianRadius;
    std::vector<int> neighborsIndexes;
    blitz::Array<std::complex<float>, 2> Z_CFIE_J;
    // constructors
    CubeArrays(void){};
    CubeArrays(const int /*cubeNumber*/, const string /*pathToReadFrom*/); 
    void copyCubeArrays (const CubeArrays& cubeArraysToCopy);
    CubeArrays(const CubeArrays&); // copy constructor
    CubeArrays& operator=(const CubeArrays&); // copy assignment operator
    ~CubeArrays();
};

CubeArrays::CubeArrays(const int cubeNumber, const string pathToReadFrom)
{
  number = cubeNumber;
  const string pathToCubeIntArrays = pathToReadFrom + intToString(cubeNumber) + "_IntArrays.txt";  
  const string pathToCube_Z = pathToReadFrom + intToString(cubeNumber);
  // reading cubeIntArrays (cube information)
  blitz::Array<int, 1> cubeIntArrays;
  blitz::ifstream ifs(pathToCubeIntArrays.c_str(), blitz::ios::binary);
  ifs.seekg (0, blitz::ios::end);
  int length = ifs.tellg();
  ifs.close();
  int N_cubeIntArrays = length/4;
  cubeIntArrays.resize(N_cubeIntArrays);
  readIntBlitzArray1DFromBinaryFile(pathToCubeIntArrays, cubeIntArrays);

  N_RWG_test = cubeIntArrays(0);
  N_RWG_src = cubeIntArrays(1);
  N_neighbors = cubeIntArrays(2);

  int startIndex = 5, stopIndex = startIndex + N_RWG_src;
  testSrc_RWGsNumbers.reserve(N_RWG_src);
  for (int index=startIndex; index<stopIndex; index++) testSrc_RWGsNumbers.push_back(cubeIntArrays(index));

  isEdgeInCartesianRadius.reserve(N_RWG_src);
  startIndex = stopIndex;
  stopIndex = startIndex + N_RWG_src;
  for (int index=startIndex; index<stopIndex; index++) isEdgeInCartesianRadius.push_back(cubeIntArrays(index));
    
  neighborsIndexes.reserve(N_neighbors);
  startIndex = stopIndex;
  stopIndex = startIndex + N_neighbors;
  for (int index=startIndex; index<stopIndex; index++) neighborsIndexes.push_back(cubeIntArrays(index));

   blitz::Array<std::complex<float>, 1> Z_CFIE_J_linear(N_RWG_test * N_RWG_src);
   readComplexFloatBlitzArray1DFromBinaryFile(pathToCube_Z, Z_CFIE_J_linear);
   Z_CFIE_J.resize(N_RWG_test, N_RWG_src);
   for (int ii=0; ii<N_RWG_test; ii++) {
     for (int jj=0; jj<N_RWG_src; jj++) Z_CFIE_J(ii, jj) = Z_CFIE_J_linear(ii*N_RWG_src + jj);
   }
}

void CubeArrays::copyCubeArrays(const CubeArrays& cubeArraysToCopy) // copy member function
{
  number = cubeArraysToCopy.number;
  N_RWG_test = cubeArraysToCopy.N_RWG_test;
  N_RWG_src = cubeArraysToCopy.N_RWG_src;
  N_neighbors = cubeArraysToCopy.N_neighbors;
  testSrc_RWGsNumbers.resize(cubeArraysToCopy.testSrc_RWGsNumbers.size());
  testSrc_RWGsNumbers = cubeArraysToCopy.testSrc_RWGsNumbers;
  isEdgeInCartesianRadius.resize(cubeArraysToCopy.isEdgeInCartesianRadius.size());
  isEdgeInCartesianRadius = cubeArraysToCopy.isEdgeInCartesianRadius;
  neighborsIndexes.resize(cubeArraysToCopy.neighborsIndexes.size());
  neighborsIndexes = cubeArraysToCopy.neighborsIndexes;
  Z_CFIE_J.resize(N_RWG_test, N_RWG_src);
  Z_CFIE_J = cubeArraysToCopy.Z_CFIE_J;
}

CubeArrays::CubeArrays(const CubeArrays& cubeArraysToCopy) // copy constructor
{
  copyCubeArrays(cubeArraysToCopy);
}

CubeArrays& CubeArrays::operator=(const CubeArrays& cubeArraysToCopy) { // copy assignment
  copyCubeArrays(cubeArraysToCopy);
  return *this;
}

CubeArrays::~CubeArrays() {
  testSrc_RWGsNumbers.clear();
  isEdgeInCartesianRadius.clear();
  neighborsIndexes.clear();
  Z_CFIE_J.free();
}

typedef std::map<int, CubeArrays> CubeArraysMap;
typedef std::map<int, CubeArrays>::const_iterator CubeArraysMapIterator;

bool frobProfilingEnabled() {
  const char* value = std::getenv("PUMA_FROB_PROFILE");
  return value != NULL && string(value) != "0";
}

struct FrobConfig {
  int max_core_rwgs;
  int overlap_rings;
};

int readNonnegativeEnvironment(const char* name, const int default_value) {
  const char* value = std::getenv(name);
  if (value == NULL || string(value).empty()) return default_value;
  errno = 0;
  char* end = NULL;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
    throw std::invalid_argument(string(name) + " must be a nonnegative integer");
  }
  return static_cast<int>(parsed);
}

FrobConfig readFrobConfig() {
  FrobConfig config;
  config.max_core_rwgs = readNonnegativeEnvironment("FROB_MAX_CORE_RWGS", 0);
  config.overlap_rings = readNonnegativeEnvironment("FROB_OVERLAP_RINGS", 1);
  return config;
}


void computeLwork(int & lwork, const int M, const int N, const int nrhs) {
  // computes the space necessary for WORK to.....work!!
  // returns the dimension of the array WORK.
  // LWORK >= max( 1, MN + max( MN, NRHS ) ).
  // For optimal performance, LWORK >= max( 1, MN + max( MN, NRHS )*NB ).
  // where MN = min(M,N) and NB is the optimum block size.
  int NB = 20;
  int mn = min(M, N);
  lwork = max( 1, mn + max( mn, nrhs )*NB );
}

void computeMyPinvCC(blitz::Array<std::complex<double>, 2>& Y, blitz::Array<std::complex<double>, 1>& A, int m, int n) {
  /* this routine computes the pseudo inverse of A
  and is a wrapper to the fortran function zgels.f
  By the way, to understand this wrapping structure, you
  better check the comments at the beginning of zgels.f */
  int lda = m;
  int ldb = max(n, m);
  int nrhs = m;
  const int N = min(ldb, nrhs);
  blitz::Array<std::complex<double>, 1> B(ldb * nrhs); // B(ldb, nrhs)
  B = 0.0;
  for (int i=0 ; i<N ; i++) B(i + i*ldb) = 1.0;
  int lwork;
  computeLwork(lwork, m, n, nrhs);
  blitz::Array<std::complex<double>, 1> work(lwork);
  work = 0.0;
  int info = 0;
  char trans = 'N';
  zgels2(trans, m, n, nrhs, A, lda, B, ldb, work, lwork, info);
  if (info==0) {
    if (m<=n) {
      Y.resize(ldb, nrhs);
      for (int j=0; j<nrhs; j++) {
        for (int i=0; i<ldb; i++) Y(i, j) = B(i + j*ldb);
      }
    }
    else {
      Y.resize(n, nrhs);
      for (int i=0; i<n; i++) {
        for (int j=0; j<nrhs; j++) Y(i, j) = B(i + j*n);
      }
    }
  }
}

void assembleLocalMatrix(blitz::Array<std::complex<float>, 2>& Z_local,
                         const int cubeNumber,
                         const CubeArraysMap & ListCubes)
{
  CubeArraysMapIterator it = ListCubes.find(cubeNumber);
  Z_local.resize((*it).second.N_RWG_src, (*it).second.N_RWG_src);
  Z_local = std::complex<float>(0.0, 0.0);
  for (int i=0; i<(*it).second.N_RWG_src; i++) Z_local(i, i) = std::complex<float>(1.0, 0.0);

  std::map<int, int> src_edges_numbers_local_src_edges_numbers;
  for (int index = 0; index<(*it).second.N_RWG_src; index++) {
    const int RWG_number = (*it).second.testSrc_RWGsNumbers[index];
    src_edges_numbers_local_src_edges_numbers[RWG_number] = index;
  }
  std::set<int> set_cubeNeighborsIndexes((*it).second.neighborsIndexes.begin(), (*it).second.neighborsIndexes.end());

  for (int index=0; index<(*it).second.N_neighbors; index++) {
    const int neighborCubeNumber = (*it).second.neighborsIndexes[index];
    CubeArraysMapIterator it_neighbor = ListCubes.find(neighborCubeNumber);
    if (index==0) {
      for (int i=0; i<(*it).second.N_RWG_test; i++) {
        for (int j=0; j<(*it).second.N_RWG_src; j++) Z_local(i, j) = (*it).second.Z_CFIE_J(i, j);
      }
    }
    else {
      std::vector<int> Z_local_lines_indexes;
      Z_local_lines_indexes.resize((*it_neighbor).second.N_RWG_test);
      for (int i=0; i<(*it_neighbor).second.N_RWG_test; i++) {
        const int RWG_number = (*it_neighbor).second.testSrc_RWGsNumbers[i];
        Z_local_lines_indexes[i] = src_edges_numbers_local_src_edges_numbers[RWG_number];
      }
      std::vector<int> common_neighborsNumbers;
      for (int i=0; i<(*it_neighbor).second.N_neighbors; i++) {
        const int val = (*it_neighbor).second.neighborsIndexes[i];
        if (set_cubeNeighborsIndexes.find(val) != set_cubeNeighborsIndexes.end()) common_neighborsNumbers.push_back(val);
      }
      std::vector<int> src_tmp;
      for (unsigned int i=0; i<common_neighborsNumbers.size(); i++) {
        const int commonNeighbor = common_neighborsNumbers[i];
        CubeArraysMapIterator it_commonNeighbor = ListCubes.find(commonNeighbor);
        for (int kk=0; kk<(*it_commonNeighbor).second.N_RWG_test; kk++) {
          src_tmp.push_back((*it_commonNeighbor).second.testSrc_RWGsNumbers[kk]);
        }
      }
      std::set<int> set_srcEdges(src_tmp.begin(), src_tmp.end());
      std::vector<int> columnsOfNeighborCubeToBeConsidered;
      for (int i=0; i<(*it_neighbor).second.N_RWG_src; i++) {
        if (set_srcEdges.find((*it_neighbor).second.testSrc_RWGsNumbers[i]) != set_srcEdges.end()) {
          columnsOfNeighborCubeToBeConsidered.push_back(i);
        }
      }
      std::vector<int> Z_local_columns_indexes;
      Z_local_columns_indexes.resize(src_tmp.size());
      for (unsigned int i=0; i<src_tmp.size(); i++) {
        const int RWG_number = src_tmp[i];
        Z_local_columns_indexes[i] = src_edges_numbers_local_src_edges_numbers[RWG_number];
      }
      for (int i=0; i<(int)Z_local_lines_indexes.size(); i++) {
        for (unsigned int j=0; j<Z_local_columns_indexes.size(); j++) {
          const int index_line = Z_local_lines_indexes[i];
          const int index_column = Z_local_columns_indexes[j];
          Z_local(index_line, index_column) = (*it_neighbor).second.Z_CFIE_J(i, columnsOfNeighborCubeToBeConsidered[j]);
        }
      }
    }
  }
}

void MgPreconditionerComputationPerCube(blitz::Array<std::complex<float>, 1>& Mg_tmp,
                                        blitz::Array<int, 1>& src_edges_numbers_2,
                                        const int cubeNumber,
                                        const CubeArraysMap & ListCubes)
{
  const bool profile = frobProfilingEnabled();
  const double assembly_start = MPI_Wtime();
  CubeArraysMapIterator it = ListCubes.find(cubeNumber);
  blitz::Array<std::complex<float>, 2> Z_local;
  assembleLocalMatrix(Z_local, cubeNumber, ListCubes);

  const double assembly_seconds = MPI_Wtime() - assembly_start;
  const double row_pack_start = MPI_Wtime();
  // further reduce the matrix size
  int N_lines = 0;
  for (int i=0; i<(*it).second.N_RWG_src; i++) N_lines += (*it).second.isEdgeInCartesianRadius[i];
  src_edges_numbers_2.resize(N_lines);
  blitz::Array<std::complex<double>, 1> Z_local_2(N_lines * (*it).second.N_RWG_src);
  Z_local_2 = 0.0;
  int line_index = 0;
  for (int i=0; i<(*it).second.N_RWG_src; i++) {
    if ((*it).second.isEdgeInCartesianRadius[i]==1) {
      for (int j=0; j<(*it).second.N_RWG_src; j++) Z_local_2(line_index + j*N_lines) = Z_local (i, j);
      src_edges_numbers_2(line_index) = (*it).second.testSrc_RWGsNumbers[i];
      line_index++;
    }
  }
  const double row_pack_seconds = MPI_Wtime() - row_pack_start;
  // compute the SAI matrix
  blitz::Array<std::complex<double>, 2> Y_CFIE;
  const double zgels_start = MPI_Wtime();
  computeMyPinvCC(Y_CFIE, Z_local_2, N_lines, (*it).second.N_RWG_src);
  const double zgels_seconds = MPI_Wtime() - zgels_start;
  
  const double output_pack_start = MPI_Wtime();
  Mg_tmp.resize((*it).second.N_RWG_test * Y_CFIE.extent(1));
  for (int i=0; i<(*it).second.N_RWG_test; i++) {
    for (int j=0; j<Y_CFIE.extent(1); j++) Mg_tmp(i*Y_CFIE.extent(1) + j) = Y_CFIE(i, j);    
  }
  const double output_pack_seconds = MPI_Wtime() - output_pack_start;

  if (profile) {
    const int m = N_lines;
    const int n = (*it).second.N_RWG_src;
    const int nrhs = m;
    const int ldb = max(n, m);
    int lwork;
    computeLwork(lwork, m, n, nrhs);
    const double qr_flop_estimate = 8.0 * min(m, n) * min(m, n) * (max(m, n) + nrhs);
    const long long workspace_estimate_bytes =
      8LL * n * n + 16LL * m * n + 16LL * ldb * nrhs +
      16LL * lwork + 16LL * n * nrhs;
    int output_nonzeros = 0;
    for (int i=0; i<(int)Mg_tmp.size(); i++) {
      if (Mg_tmp(i) != std::complex<float>(0.0, 0.0)) output_nonzeros++;
    }
    std::ostringstream record;
    record << "FROB_PROFILE"
           << " rank=" << MPI::COMM_WORLD.Get_rank()
           << " cube=" << cubeNumber
           << " m=" << m
           << " n=" << n
           << " nrhs=" << nrhs
           << " neighbors=" << (*it).second.N_neighbors
           << " selected_rows=" << N_lines
           << " qr_flop_estimate=" << qr_flop_estimate
           << " workspace_estimate_bytes=" << workspace_estimate_bytes
           << " output_nonzeros=" << output_nonzeros
           << " assembly_s=" << assembly_seconds
           << " row_pack_s=" << row_pack_seconds
           << " zgels_s=" << zgels_seconds
           << " output_pack_s=" << output_pack_seconds;
    std::cout << record.str() << std::endl;
  }
}

struct BoundedCubePreconditioner {
  std::vector<std::complex<float> > values;
  std::vector<int> source_rwgs;
  std::vector<int> row_source_starts;
  std::vector<int> row_source_stops;
};

void loadGlobalRwgTriangles(
    blitz::Array<int, 2>& rwg_signed_triangles,
    const string& simu_dir)
{
  const int my_id = MPI::COMM_WORLD.Get_rank();
  int number_of_rwgs = 0;
  int valid = 1;
  const string mesh_path = simu_dir + "/tmp0/mesh/";
  const string triangles_path = mesh_path + "RWGNumber_signedTriangles.txt";
  if (my_id == 0) {
    readIntFromASCIIFile(mesh_path + "N_RWG.txt", number_of_rwgs);
    std::ifstream stream(triangles_path.c_str(), std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(2LL * number_of_rwgs * sizeof(int))) {
      valid = 0;
    }
  }
  MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&number_of_rwgs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (!valid || number_of_rwgs <= 0) {
    throw std::runtime_error("invalid RWGNumber_signedTriangles topology file");
  }
  rwg_signed_triangles.resize(number_of_rwgs, 2);
  if (my_id == 0) {
    readIntBlitzArray2DFromBinaryFile(triangles_path, rwg_signed_triangles);
  }
  MPI_Bcast(rwg_signed_triangles.data(), 2 * number_of_rwgs, MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<std::vector<int> > buildLocalAdjacency(
    const CubeArrays& cube,
    const blitz::Array<int, 2>& rwg_signed_triangles) {
  std::map<int, std::vector<int> > triangle_members;
  for (int local_index = 0; local_index < cube.N_RWG_src; local_index++) {
    const int global_rwg = cube.testSrc_RWGsNumbers[local_index];
    if (global_rwg < 0 || global_rwg >= rwg_signed_triangles.extent(0)) {
      throw std::runtime_error("cube references an RWG outside the topology table");
    }
    triangle_members[rwg_signed_triangles(global_rwg, 0)].push_back(local_index);
    const int second_triangle = rwg_signed_triangles(global_rwg, 1);
    if (second_triangle != rwg_signed_triangles(global_rwg, 0)) {
      triangle_members[second_triangle].push_back(local_index);
    }
  }

  std::vector<std::set<int> > adjacency_sets(cube.N_RWG_src);
  for (std::map<int, std::vector<int> >::const_iterator bucket = triangle_members.begin();
       bucket != triangle_members.end();
       bucket++) {
    const std::vector<int>& members = bucket->second;
    for (unsigned int first = 0; first < members.size(); first++) {
      for (unsigned int second = first + 1; second < members.size(); second++) {
        adjacency_sets[members[first]].insert(members[second]);
        adjacency_sets[members[second]].insert(members[first]);
      }
    }
  }

  std::vector<std::vector<int> > adjacency(cube.N_RWG_src);
  for (int local_index = 0; local_index < cube.N_RWG_src; local_index++) {
    adjacency[local_index].assign(
      adjacency_sets[local_index].begin(), adjacency_sets[local_index].end());
  }
  return adjacency;
}

BoundedCubePreconditioner MgBoundedPreconditionerComputationPerCube(
    const int cubeNumber,
    const CubeArraysMap & ListCubes,
    const FrobConfig& config,
    const blitz::Array<int, 2>& rwg_signed_triangles)
{
  const bool profile = frobProfilingEnabled();
  const double assembly_start = MPI_Wtime();
  CubeArraysMapIterator it = ListCubes.find(cubeNumber);
  blitz::Array<std::complex<float>, 2> Z_local;
  assembleLocalMatrix(Z_local, cubeNumber, ListCubes);
  const double assembly_seconds = MPI_Wtime() - assembly_start;

  const std::vector<std::vector<int> > adjacency = buildLocalAdjacency(
    (*it).second, rwg_signed_triangles);
  const std::vector<FrobBlock> blocks = buildFrobBlocks(
    adjacency,
    (*it).second.N_RWG_test,
    config.max_core_rwgs,
    config.overlap_rings);

  BoundedCubePreconditioner result;
  result.row_source_starts.assign((*it).second.N_RWG_test, -1);
  result.row_source_stops.assign((*it).second.N_RWG_test, -1);
  std::vector<std::vector<std::complex<float> > > row_values((*it).second.N_RWG_test);

  for (unsigned int block_number = 0; block_number < blocks.size(); block_number++) {
    const FrobBlock& block = blocks[block_number];
    std::vector<int> selected_rows;
    for (unsigned int support_position = 0;
         support_position < block.support_indexes.size();
         support_position++) {
      const int local_index = block.support_indexes[support_position];
      if ((*it).second.isEdgeInCartesianRadius[local_index] == 1) {
        selected_rows.push_back(local_index);
      }
    }
    if (selected_rows.empty()) {
      throw std::runtime_error("bounded FROB block has no selected source rows");
    }

    const int matrix_rows = selected_rows.size();
    const int matrix_columns = block.support_indexes.size();
    blitz::Array<std::complex<double>, 1> Z_block(matrix_rows * matrix_columns);
    for (int column = 0; column < matrix_columns; column++) {
      const int local_column = block.support_indexes[column];
      for (int row = 0; row < matrix_rows; row++) {
        Z_block(row + column * matrix_rows) = Z_local(selected_rows[row], local_column);
      }
    }

    const int source_start = result.source_rwgs.size();
    for (int row = 0; row < matrix_rows; row++) {
      result.source_rwgs.push_back((*it).second.testSrc_RWGsNumbers[selected_rows[row]]);
    }
    const int source_stop = result.source_rwgs.size();

    blitz::Array<std::complex<double>, 2> Y_CFIE;
    const double zgels_start = MPI_Wtime();
    computeMyPinvCC(Y_CFIE, Z_block, matrix_rows, matrix_columns);
    const double zgels_seconds = MPI_Wtime() - zgels_start;

    for (unsigned int core_position = 0;
         core_position < block.core_indexes.size();
         core_position++) {
      const int core_index = block.core_indexes[core_position];
      int support_position = -1;
      for (unsigned int candidate = 0;
           candidate < block.support_indexes.size();
           candidate++) {
        if (block.support_indexes[candidate] == core_index) {
          support_position = candidate;
          break;
        }
      }
      if (support_position < 0) {
        throw std::runtime_error("bounded FROB core row is absent from its support");
      }
      result.row_source_starts[core_index] = source_start;
      result.row_source_stops[core_index] = source_stop;
      row_values[core_index].reserve(matrix_rows);
      for (int row = 0; row < matrix_rows; row++) {
        row_values[core_index].push_back(
          static_cast<std::complex<float> >(Y_CFIE(support_position, row)));
      }
    }

    if (profile) {
      std::ostringstream record;
      record << "FROB_BLOCK_PROFILE"
             << " rank=" << MPI::COMM_WORLD.Get_rank()
             << " cube=" << cubeNumber
             << " block=" << block_number
             << " core_rwgs=" << block.core_indexes.size()
             << " support_rwgs=" << matrix_columns
             << " selected_rows=" << matrix_rows
             << " assembly_s=" << assembly_seconds
             << " zgels_s=" << zgels_seconds;
      std::cout << record.str() << std::endl;
    }
  }

  int expected_values = 0;
  for (int core_index = 0; core_index < (*it).second.N_RWG_test; core_index++) {
    if (result.row_source_starts[core_index] < 0 || result.row_source_stops[core_index] < 0) {
      throw std::runtime_error("bounded FROB did not assign every core row");
    }
    if (result.row_source_starts[core_index] > result.row_source_stops[core_index] ||
        result.row_source_stops[core_index] > static_cast<int>(result.source_rwgs.size())) {
      throw std::runtime_error("bounded FROB produced an invalid source range");
    }
    expected_values += result.row_source_stops[core_index] - result.row_source_starts[core_index];
    result.values.insert(
      result.values.end(),
      row_values[core_index].begin(),
      row_values[core_index].end());
  }
  if (expected_values != static_cast<int>(result.values.size())) {
    throw std::runtime_error("bounded FROB coefficient count does not match row ranges");
  }
  return result;
}

int main(int argc, char* argv[]) {

  MPI::Init();
  const int my_id = MPI::COMM_WORLD.Get_rank();
  FrobConfig frob_config;
  try {
    frob_config = readFrobConfig();
  }
  catch (const std::exception& error) {
    if (my_id == 0) std::cerr << "FROB_CONFIG_ERROR " << error.what() << std::endl;
    MPI::Finalize();
    return 2;
  }
  if (my_id == 0) {
    std::cout << "FROB_CONFIG mode="
              << (frob_config.max_core_rwgs > 0 ? "bounded" : "legacy")
              << " max_core_rwgs=" << frob_config.max_core_rwgs
              << " overlap_rings=" << frob_config.overlap_rings
              << std::endl;
  }

  string simuDir = ".";
  if ( argc > 2 ) {
    if( string(argv[1]) == "--simudir" ) simuDir = argv[2];
  }

  // general variables
  const string SIMU_DIR = simuDir;
  const string TMP = SIMU_DIR + "/tmp" + intToString(my_id);
  const string Z_TMP_DATA_PATH = TMP + "/Z_tmp/";
  const string SAI_PRECOND_DATA_PATH = TMP + "/Mg_LeftFrob/";
  const string OCTTREE_DATA_PATH = TMP + "/octtree_data/";
  string filename;

  blitz::Array<int, 2> rwg_signed_triangles;
  if (frob_config.max_core_rwgs > 0) {
    try {
      loadGlobalRwgTriangles(rwg_signed_triangles, SIMU_DIR);
    }
    catch (const std::exception& error) {
      if (my_id == 0) std::cerr << "FROB_TOPOLOGY_ERROR " << error.what() << std::endl;
      MPI::Finalize();
      return 2;
    }
  }

  blitz::Array<int, 1> chunkNumbers, cubeNumber_to_chunkNumber;
  readIntBlitzArray1DFromASCIIFile(SAI_PRECOND_DATA_PATH + "chunkNumbers.txt", chunkNumbers);
  readIntBlitzArray1DFromASCIIFile(SAI_PRECOND_DATA_PATH + "cubeNumber_to_chunkNumber.txt", cubeNumber_to_chunkNumber);

  const int N_chunks = chunkNumbers.size();
  for (int i=0; i<N_chunks; i++) {
    const int chunk = chunkNumbers(i);
    blitz::Array<int, 1> cubesNumbers;
    readIntBlitzArray1DFromASCIIFile(SAI_PRECOND_DATA_PATH + "chunk" + intToString(chunk) + "cubesNumbers.txt", cubesNumbers);
    const int N_cubes = cubesNumbers.size();
    CubeArraysMap ListCubes;
    // variables needed later
    int N_RWG = 0, N_precond = 0, N_q_array = 0;
    std::vector<int> N_ColumnsPerCube;
    N_ColumnsPerCube.resize(N_cubes);
    // we construct a list of cubes
    for (int j=0; j<N_cubes; j++) {
      const int cubeNumber = cubesNumbers(j);
      // we read the cube data and construct the cube
      const string pathToCube = Z_TMP_DATA_PATH + "chunk" + intToString(chunk) + "/";
      const CubeArrays cube(cubeNumber, pathToCube);
      // we add the cube only if it is not in the list
      if (ListCubes.find(cubeNumber) == ListCubes.end()) ListCubes.insert(CubeArraysMap::value_type(cubeNumber, cube));
      // we then add the neighbor cubes to the list
      for (int kk=0; kk<cube.N_neighbors; kk++) {
        const int neighborCubeNumber = cube.neighborsIndexes[kk];
        if (ListCubes.find(neighborCubeNumber) == ListCubes.end()) {
          const int chunkNumberNeighbor = cubeNumber_to_chunkNumber(neighborCubeNumber);
          const string pathToCubeNeighbor = Z_TMP_DATA_PATH + "chunk" + intToString(chunkNumberNeighbor) + "/";
          const CubeArrays cubeNeighbor(neighborCubeNumber, pathToCubeNeighbor);
          ListCubes.insert(CubeArraysMap::value_type(neighborCubeNumber, cubeNeighbor));
        }
      }
      // then we need to construct a series of indexes and offsets
      N_RWG += cube.N_RWG_test;
      int N_isEdgeInCartesianRadius = 0;
      for (unsigned int kk=0; kk<cube.isEdgeInCartesianRadius.size(); kk++) N_isEdgeInCartesianRadius += cube.isEdgeInCartesianRadius[kk];
      N_ColumnsPerCube[j] = N_isEdgeInCartesianRadius;
      // N_precond = number of elements in the preconditioner chunk
      N_precond += N_isEdgeInCartesianRadius * cube.N_RWG_test;
      N_q_array += N_isEdgeInCartesianRadius;      
    }
    // constructing test_RWG_numbers
    blitz::Array<int, 1> test_RWG_numbers(N_RWG);
    int startIndexInRWGNumbers = 0;
    for (int j=0; j<N_cubes; j++) {
      const int cubeNumber = cubesNumbers(j);
      CubeArraysMapIterator it = ListCubes.find(cubeNumber);
      const CubeArrays cube((*it).second);
      for (int kk=0; kk<cube.N_RWG_test; kk++) test_RWG_numbers(kk + startIndexInRWGNumbers) = cube.testSrc_RWGsNumbers[kk];
      startIndexInRWGNumbers += cube.N_RWG_test;
    }
    if (frob_config.max_core_rwgs > 0) {
      std::vector<BoundedCubePreconditioner> bounded_results;
      int bounded_N_precond = 0;
      int bounded_N_q_array = 0;
      for (int j=0; j<N_cubes; j++) {
        const BoundedCubePreconditioner result = MgBoundedPreconditionerComputationPerCube(
          cubesNumbers(j), ListCubes, frob_config, rwg_signed_triangles);
        bounded_N_precond += static_cast<int>(result.values.size());
        bounded_N_q_array += static_cast<int>(result.source_rwgs.size());
        bounded_results.push_back(result);
      }

      blitz::Array<std::complex<float>, 1> Mg(bounded_N_precond);
      blitz::Array<int, 2> rowIndexToColumnIndexes(N_RWG, 2);
      blitz::Array<int, 1> src_RWG_numbers(bounded_N_q_array);
      int value_offset = 0;
      int source_offset = 0;
      int row_offset = 0;
      for (int j=0; j<N_cubes; j++) {
        const BoundedCubePreconditioner& result = bounded_results[j];
        for (int value_index = 0; value_index < static_cast<int>(result.values.size()); value_index++) {
          Mg(value_offset++) = result.values[value_index];
        }
        for (int source_index = 0; source_index < static_cast<int>(result.source_rwgs.size()); source_index++) {
          src_RWG_numbers(source_offset + source_index) = result.source_rwgs[source_index];
        }
        for (int row = 0; row < static_cast<int>(result.row_source_starts.size()); row++) {
          rowIndexToColumnIndexes(row_offset + row, 0) = source_offset + result.row_source_starts[row];
          rowIndexToColumnIndexes(row_offset + row, 1) = source_offset + result.row_source_stops[row];
        }
        source_offset += static_cast<int>(result.source_rwgs.size());
        row_offset += static_cast<int>(result.row_source_starts.size());
      }
      if (value_offset != bounded_N_precond ||
          source_offset != bounded_N_q_array ||
          row_offset != N_RWG) {
        throw std::runtime_error("bounded FROB chunk packing count mismatch");
      }
      writeComplexFloatBlitzArray1DToBinaryFile(SAI_PRECOND_DATA_PATH + "Mg_LeftFrob"+ intToString(chunk) + ".txt", Mg);
      writeIntBlitzArray1DToBinaryFile(SAI_PRECOND_DATA_PATH + "src_RWG_numbers" + intToString(chunk) + ".txt", src_RWG_numbers);
      writeIntBlitzArray2DToBinaryFile(SAI_PRECOND_DATA_PATH + "rowIndexToColumnIndexes" + intToString(chunk) + ".txt", rowIndexToColumnIndexes);
      writeIntBlitzArray1DToBinaryFile(SAI_PRECOND_DATA_PATH + "test_RWG_numbers" + intToString(chunk) + ".txt", test_RWG_numbers);
      writeIntToASCIIFile(SAI_PRECOND_DATA_PATH + "N_test_RWG" + intToString(chunk) + ".txt", test_RWG_numbers.size());
      writeIntToASCIIFile(SAI_PRECOND_DATA_PATH + "N_near" + intToString(chunk) + ".txt", bounded_N_precond);
      writeIntToASCIIFile(SAI_PRECOND_DATA_PATH + "N_src_RWG" + intToString(chunk) + ".txt", src_RWG_numbers.size());
      continue;
    }
    // calculation of the SAI preconditioner
    // for the q_array, each src function for all the testing functions of a cube appears only once
    // instead of once per testing function. This allows a dramatic reduction in q_array.size
    blitz::Array<std::complex<float>, 1> Mg(N_precond); // N_precond = number of elements in the preconditioner chunk
    blitz::Array<int, 2> rowIndexToColumnIndexes(N_RWG, 2);
    blitz::Array<int, 1> src_RWG_numbers(N_q_array);
    int startIndex = 0, startIndexInQArray = 0;
    int indexN_ColumnsPerCube = 0, index_in_rowIndexToColumnIndexes = 0;
    for (int j=0; j<N_cubes; j++) {
      const int cubeNumber = cubesNumbers(j);
      blitz::Array<int, 1> Mg_q_array;
      blitz::Array<std::complex<float>, 1> Mg_tmp;
      MgPreconditionerComputationPerCube(Mg_tmp, Mg_q_array, cubeNumber, ListCubes);
      // filling Mg
      for (unsigned int kk=0; kk<Mg_tmp.size(); kk++) Mg(kk + startIndex) = Mg_tmp(kk);
      startIndex += Mg_tmp.size();
      // filling src_RWG_numbers
      for (unsigned int kk=0; kk<Mg_q_array.size(); kk++) src_RWG_numbers(kk + startIndexInQArray) = Mg_q_array(kk);
      // the rest
      CubeArraysMapIterator it = ListCubes.find(cubeNumber);
      const int indInf = index_in_rowIndexToColumnIndexes;
      const int indSup = index_in_rowIndexToColumnIndexes + (*it).second.N_RWG_test;
      for (int kk=0; kk<(*it).second.N_RWG_test; kk++) {
        rowIndexToColumnIndexes(kk + indInf, 0) = startIndexInQArray;
        rowIndexToColumnIndexes(kk + indInf, 1) = startIndexInQArray + N_ColumnsPerCube[indexN_ColumnsPerCube];
      }
      index_in_rowIndexToColumnIndexes = indSup;
      startIndexInQArray += N_ColumnsPerCube[indexN_ColumnsPerCube];
      indexN_ColumnsPerCube += 1;  
    }
    // we write the arrays to disk
    writeComplexFloatBlitzArray1DToBinaryFile(SAI_PRECOND_DATA_PATH + "Mg_LeftFrob"+ intToString(chunk) + ".txt", Mg);
    writeIntBlitzArray1DToBinaryFile(SAI_PRECOND_DATA_PATH + "src_RWG_numbers" + intToString(chunk) + ".txt", src_RWG_numbers);
    writeIntBlitzArray2DToBinaryFile(SAI_PRECOND_DATA_PATH + "rowIndexToColumnIndexes" + intToString(chunk) + ".txt", rowIndexToColumnIndexes);
    writeIntBlitzArray1DToBinaryFile(SAI_PRECOND_DATA_PATH + "test_RWG_numbers" + intToString(chunk) + ".txt", test_RWG_numbers);
    // now we write the scalar values
    writeIntToASCIIFile(SAI_PRECOND_DATA_PATH + "N_test_RWG" + intToString(chunk) + ".txt", test_RWG_numbers.size());
    writeIntToASCIIFile(SAI_PRECOND_DATA_PATH + "N_near" + intToString(chunk) + ".txt", N_precond);
    writeIntToASCIIFile(SAI_PRECOND_DATA_PATH + "N_src_RWG" + intToString(chunk) + ".txt", src_RWG_numbers.size());
  }
  
  // Get peak memory usage of each rank
  float memusage_local_MB = static_cast<float>(MemoryUsageGetPeak())/(1024.0*1024.0);
  float tot_memusage_MB, max_memusage_MB;
  MPI_Reduce(&memusage_local_MB, &max_memusage_MB, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&memusage_local_MB, &tot_memusage_MB, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  int num_procs = MPI::COMM_WORLD.Get_size();
  if (my_id==0) {
    std::cout << "MEMINFO " << argv[0] << " : max mem = " << max_memusage_MB << " MB, tot mem = " << tot_memusage_MB << " MB, avg mem = " << tot_memusage_MB/num_procs << " MB" << std::endl;
  }
  MPI::Finalize();
  return 0;
}
