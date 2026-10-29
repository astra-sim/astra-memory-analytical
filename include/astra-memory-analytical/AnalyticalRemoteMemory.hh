// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include <astra-sim/common/AstraRemoteMemoryAPI.hh>
#include <astra-sim/system/common/Callable.hh>
#include <astra-sim/system/common/Sys.hh>

namespace Analytical {
enum MemoryArchitectureType {
    NO_MEMORY_EXPANSION = 0,
    PER_NODE_MEMORY_EXPANSION,
    PER_NPU_MEMORY_EXPANSION,
    MEMORY_POOL
};

class PendingMemoryRequest {
  public:
    PendingMemoryRequest(uint64_t tensor_size,
                         AstraSim::WorkloadLayerHandlerData* wlhd)
        : tensor_size{tensor_size},
          wlhd{wlhd} {}

    uint64_t tensor_size;
    AstraSim::WorkloadLayerHandlerData* wlhd;
};

class AnalyticalRemoteMemory : public AstraSim::AstraRemoteMemoryAPI,
                               public AstraSim::Callable {
  public:
    explicit AnalyticalRemoteMemory(
        const std::string& memory_configuration) noexcept;
    void set_sys(int id, AstraSim::Sys* sys) override;
    void issue(uint64_t tensor_size,
               AstraSim::WorkloadLayerHandlerData* wlhd) override;
    void call(AstraSim::EventType type, AstraSim::CallData* data) override;

  private:
    uint64_t get_remote_mem_runtime(uint64_t tensor_size) const;

    MemoryArchitectureType mem_type = NO_MEMORY_EXPANSION;
    uint64_t remote_mem_latency = 0;  // remote memory access latency in nanosec
    uint64_t remote_mem_bw = 0;       // remote memory bandwidth in GiB/sec
    std::vector<bool> ongoing_transaction;

    // per-node memory expansion
    int num_nodes = 0;
    int num_npus_per_node = 0;

    std::unordered_map<int, AstraSim::Sys*> sys_map;
    std::vector<std::deque<PendingMemoryRequest>> pending_requests;
};
}  // namespace Analytical
