// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

#include <astra-memory-analytical/AnalyticalRemoteMemory.hh>

#include <json/json.hpp>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <astra-sim/common/Common.hh>
#include <astra-sim/system/common/WorkloadLayerHandlerData.hh>
#include <astra-sim/workload/Workload.hh>

using namespace AstraSim;
using namespace Analytical;
using json = nlohmann::json;

AnalyticalRemoteMemory::AnalyticalRemoteMemory(
    const std::string& memory_configuration) noexcept {
    auto conf_file = std::ifstream{};

    conf_file.open(memory_configuration);
    if (!conf_file) {
        std::cerr << "Unable to open file: " << memory_configuration
                  << std::endl;
        exit(1);
    }

    auto j = json{};
    try {
        conf_file >> j;
    } catch (const json::exception& e) {
        std::cerr << "Malformed remote memory configuration file "
                  << memory_configuration << ": " << e.what() << std::endl;
        exit(1);
    }

    if (j.contains("memory-type")) {
        const std::string mem_type_str = j["memory-type"];
        if (mem_type_str.compare("NO_MEMORY_EXPANSION") == 0) {
            mem_type = NO_MEMORY_EXPANSION;
        } else if (mem_type_str.compare("PER_NODE_MEMORY_EXPANSION") == 0) {
            mem_type = PER_NODE_MEMORY_EXPANSION;
        } else if (mem_type_str.compare("PER_NPU_MEMORY_EXPANSION") == 0) {
            mem_type = PER_NPU_MEMORY_EXPANSION;
        } else if (mem_type_str.compare("MEMORY_POOL") == 0) {
            mem_type = MEMORY_POOL;
        } else {
            std::cerr << "Unsupported memory type: " << mem_type_str
                      << std::endl;
            exit(1);
        }
    }

    if (mem_type == PER_NODE_MEMORY_EXPANSION) {
        if (j.contains("num-nodes")) {
            num_nodes = j["num-nodes"];
        }
        if (j.contains("num-npus-per-node")) {
            num_npus_per_node = j["num-npus-per-node"];
        }
    }

    if (j.contains("remote-mem-latency")) {
        remote_mem_latency = j["remote-mem-latency"];
    }

    if (j.contains("remote-mem-bw")) {
        remote_mem_bw = j["remote-mem-bw"];
    }

    if (mem_type == PER_NODE_MEMORY_EXPANSION) {
        for (int i = 0; i < num_nodes; i++) {
            ongoing_transaction.push_back(false);
            const auto dpmr = std::deque<PendingMemoryRequest>{};
            pending_requests.push_back(dpmr);
        }
    } else if (mem_type == MEMORY_POOL) {
        ongoing_transaction.push_back(false);
        const auto dpmr = std::deque<PendingMemoryRequest>{};
        pending_requests.push_back(dpmr);
    }

    conf_file.close();
}

void AnalyticalRemoteMemory::set_sys(const int id, Sys* const sys) {
    sys_map[id] = sys;
}

void AnalyticalRemoteMemory::issue(const uint64_t tensor_size,
                                   WorkloadLayerHandlerData* const wlhd) {
    const auto sys_id = wlhd->sys_id;

    if (mem_type == NO_MEMORY_EXPANSION) {
        std::cerr
            << "Remote memory access is not supported in NO_MEMORY_EXPANSION"
            << std::endl;
        exit(1);
    } else if (mem_type == PER_NODE_MEMORY_EXPANSION) {
        const auto node_id = sys_id / num_npus_per_node;
        if (ongoing_transaction[node_id]) {
            const auto pmr = PendingMemoryRequest{tensor_size, wlhd};
            pending_requests[node_id].push_back(pmr);
        } else {
            const auto runtime = get_remote_mem_runtime(tensor_size);

            auto* const sys = sys_map[sys_id];
            sys->register_event(wlhd->workload, EventType::General, wlhd,
                                runtime);

            sys->register_event(this, EventType::General, wlhd, runtime);

            ongoing_transaction[node_id] = true;
        }
    } else if (mem_type == PER_NPU_MEMORY_EXPANSION) {
        const auto runtime = get_remote_mem_runtime(tensor_size);
        auto* const sys = sys_map[sys_id];
        sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);
    } else if (mem_type == MEMORY_POOL) {
        if (ongoing_transaction[0]) {
            const auto pmr = PendingMemoryRequest{tensor_size, wlhd};
            pending_requests[0].push_back(pmr);
        } else {
            const auto runtime = get_remote_mem_runtime(tensor_size);

            auto* const sys = sys_map[sys_id];
            sys->register_event(wlhd->workload, EventType::General, wlhd,
                                runtime);

            sys->register_event(this, EventType::General, wlhd, runtime);

            ongoing_transaction[0] = true;
        }
    }
}

void AnalyticalRemoteMemory::call(EventType /*type*/, CallData* const data) {
    if (mem_type == PER_NODE_MEMORY_EXPANSION) {
        auto* const wlhd = dynamic_cast<WorkloadLayerHandlerData*>(data);
        assert(wlhd);
        const auto node_id = wlhd->sys_id / num_npus_per_node;
        if (!pending_requests[node_id].empty()) {
            const auto pmr = pending_requests[node_id].front();
            pending_requests[node_id].pop_front();

            const auto runtime = get_remote_mem_runtime(pmr.tensor_size);

            auto* const sys = sys_map[pmr.wlhd->sys_id];
            sys->register_event(pmr.wlhd->workload, EventType::General,
                                pmr.wlhd, runtime);

            sys->register_event(this, EventType::General, pmr.wlhd, runtime);

            ongoing_transaction[node_id] = true;
        } else {
            ongoing_transaction[node_id] = false;
        }
    } else if (mem_type == MEMORY_POOL) {
        if (!pending_requests[0].empty()) {
            const auto pmr = pending_requests[0].front();
            pending_requests[0].pop_front();

            const auto runtime = get_remote_mem_runtime(pmr.tensor_size);

            auto* const sys = sys_map[pmr.wlhd->sys_id];
            sys->register_event(pmr.wlhd->workload, EventType::General,
                                pmr.wlhd, runtime);

            sys->register_event(this, EventType::General, pmr.wlhd, runtime);

            ongoing_transaction[0] = true;
        } else {
            ongoing_transaction[0] = false;
        }
    }
}

uint64_t AnalyticalRemoteMemory::get_remote_mem_runtime(
    const uint64_t tensor_size) const {
    // remote_mem_latency is in nanosec; remote_mem_bw is in GiB/sec.
    // Convert to the simulator's picosecond tick granularity.
    const auto bw_bytes_per_ps = static_cast<double>(remote_mem_bw) *
                                 static_cast<double>(1ULL << 30) / 1e12;
    const auto runtime =
        remote_mem_latency * 1000 +
        static_cast<uint64_t>(static_cast<double>(tensor_size) /
                              bw_bytes_per_ps);
    return runtime;
}
