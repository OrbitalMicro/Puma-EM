#ifndef FROB_BLOCKS_H
#define FROB_BLOCKS_H

#include <stdexcept>
#include <vector>

struct FrobBlock {
  std::vector<int> core_indexes;
  std::vector<int> support_indexes;
};

inline std::vector<FrobBlock> buildFrobBlocks(
    const std::vector<std::vector<int> >& adjacency,
    const int core_count,
    const int max_core_rwgs,
    const int overlap_rings) {
  if (core_count < 0 || core_count > static_cast<int>(adjacency.size())) {
    throw std::invalid_argument("core_count is outside the adjacency graph");
  }
  if (max_core_rwgs <= 0) {
    throw std::invalid_argument("max_core_rwgs must be positive");
  }
  if (overlap_rings < 0) {
    throw std::invalid_argument("overlap_rings must be nonnegative");
  }

  std::vector<bool> owned(core_count, false);
  std::vector<FrobBlock> blocks;
  int owned_count = 0;

  while (owned_count < core_count) {
    FrobBlock block;
    int seed = 0;
    while (seed < core_count && owned[seed]) seed++;
    block.core_indexes.push_back(seed);
    owned[seed] = true;
    owned_count++;

    while (static_cast<int>(block.core_indexes.size()) < max_core_rwgs &&
           owned_count < core_count) {
      int next_index = -1;
      for (int candidate = 0; candidate < core_count && next_index < 0; candidate++) {
        if (owned[candidate]) continue;
        for (unsigned int core_position = 0;
             core_position < block.core_indexes.size() && next_index < 0;
             core_position++) {
          const int core_index = block.core_indexes[core_position];
          const std::vector<int>& neighbors = adjacency[core_index];
          for (unsigned int neighbor_position = 0;
               neighbor_position < neighbors.size();
               neighbor_position++) {
            if (neighbors[neighbor_position] == candidate) {
              next_index = candidate;
              break;
            }
          }
        }
      }
      if (next_index < 0) {
        for (int candidate = 0; candidate < core_count; candidate++) {
          if (!owned[candidate]) {
            next_index = candidate;
            break;
          }
        }
      }
      block.core_indexes.push_back(next_index);
      owned[next_index] = true;
      owned_count++;
    }

    std::vector<bool> in_support(adjacency.size(), false);
    for (unsigned int core_position = 0;
         core_position < block.core_indexes.size();
         core_position++) {
      in_support[block.core_indexes[core_position]] = true;
    }
    for (int ring = 0; ring < overlap_rings; ring++) {
      std::vector<bool> expanded(in_support);
      for (unsigned int index = 0; index < adjacency.size(); index++) {
        if (!in_support[index]) continue;
        const std::vector<int>& neighbors = adjacency[index];
        for (unsigned int neighbor_position = 0;
             neighbor_position < neighbors.size();
             neighbor_position++) {
          const int neighbor = neighbors[neighbor_position];
          if (neighbor < 0 || neighbor >= static_cast<int>(adjacency.size())) {
            throw std::invalid_argument("adjacency index is outside the graph");
          }
          expanded[neighbor] = true;
        }
      }
      in_support.swap(expanded);
    }
    for (unsigned int index = 0; index < in_support.size(); index++) {
      if (in_support[index]) block.support_indexes.push_back(index);
    }
    blocks.push_back(block);
  }

  return blocks;
}

#endif