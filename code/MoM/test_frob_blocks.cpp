#include <cassert>
#include <stdexcept>
#include <vector>

#include "frob_blocks.h"

int main() {
  std::vector<std::vector<int> > adjacency(7);
  adjacency[0].push_back(1);
  adjacency[1].push_back(0);
  adjacency[1].push_back(2);
  adjacency[2].push_back(1);
  adjacency[2].push_back(3);
  adjacency[3].push_back(2);
  adjacency[3].push_back(4);
  adjacency[4].push_back(3);
  adjacency[4].push_back(5);
  adjacency[5].push_back(4);
  adjacency[5].push_back(6);
  adjacency[6].push_back(5);

  const std::vector<FrobBlock> blocks = buildFrobBlocks(adjacency, 5, 2, 1);
  assert(blocks.size() == 3);
  assert(blocks[0].core_indexes.size() == 2);
  assert(blocks[0].core_indexes[0] == 0);
  assert(blocks[0].core_indexes[1] == 1);
  assert(blocks[0].support_indexes.size() == 3);
  assert(blocks[0].support_indexes[2] == 2);
  assert(blocks[1].core_indexes[0] == 2);
  assert(blocks[1].core_indexes[1] == 3);
  assert(blocks[1].support_indexes.size() == 4);
  assert(blocks[1].support_indexes[0] == 1);
  assert(blocks[1].support_indexes[3] == 4);
  assert(blocks[2].core_indexes.size() == 1);
  assert(blocks[2].core_indexes[0] == 4);

  std::vector<std::vector<int> > disconnected(4);
  const std::vector<FrobBlock> disconnected_blocks =
    buildFrobBlocks(disconnected, 4, 3, 0);
  assert(disconnected_blocks.size() == 2);
  assert(disconnected_blocks[0].core_indexes.size() == 3);
  assert(disconnected_blocks[1].core_indexes.size() == 1);

  bool rejected_invalid_neighbor = false;
  disconnected[0].push_back(4);
  try {
    buildFrobBlocks(disconnected, 4, 2, 1);
  }
  catch (const std::invalid_argument&) {
    rejected_invalid_neighbor = true;
  }
  assert(rejected_invalid_neighbor);

  return 0;
}