/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
 *******************************************************************************/

#include "extern/memory_backend/analytical/AnalyticalMemory.hh"

#include <fstream>
#include <iostream>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/json.hpp"

using namespace std;
using namespace AstraSim;
using namespace Analytical;
using json = nlohmann::json;

AnalyticalMemory::AnalyticalMemory(string memory_configuration) noexcept {
  ifstream conf_file;

  conf_file.open(memory_configuration);
  if (!conf_file) {
    cerr << "Unable to open file: " << memory_configuration << endl;
    exit(1);
  }

  json j;
  conf_file >> j;

  if (j.contains("memory-type")) {
    string mem_type_str = j["memory-type"];
    if (mem_type_str.compare("NO_MEMORY_EXPANSION") == 0) {
      mem_type = NO_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NODE_MEMORY_EXPANSION") == 0) {
      mem_type = PER_NODE_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NPU_MEMORY_EXPANSION") == 0) {
      mem_type = PER_NPU_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("MEMORY_POOL") == 0) {
      mem_type = MEMORY_POOL;
    } else {
      cerr << "Unsupported memory type: " << mem_type_str << endl;
      exit(1);
    }
  }

  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    num_nodes = 0;
    if (j.contains("num-nodes")) {
      num_nodes = j["num-nodes"];
    }
    num_npus_per_node = 0;
    if (j.contains("num-npus-per-node")) {
      num_npus_per_node = j["num-npus-per-node"];
    }
  }

  local_mem_latency = 0;
  if (j.contains("local-mem-latency")) {
    local_mem_latency = j["local-mem-latency"];
  }

  local_mem_bw = 0;
  if (j.contains("local-mem-bw")) {
    local_mem_bw = j["local-mem-bw"];
    local_mem_bw = local_mem_bw * 1000000000; // GB/sec
  }

  remote_mem_latency = 0;
  if (j.contains("remote-mem-latency")) {
    remote_mem_latency = j["remote-mem-latency"];
  }

  remote_mem_bw = 0;
  if (j.contains("remote-mem-bw")) {
    remote_mem_bw = j["remote-mem-bw"];
    remote_mem_bw = remote_mem_bw * 1000000000; // GB/sec
  }

  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    for (int i = 0; i < num_nodes; i++) {
      ongoing_transaction.push_back(false);
      deque<PendingMemoryRequest> dpmr;
      pending_requests.push_back(dpmr);
    }
  } else if (mem_type == MEMORY_POOL) {
    ongoing_transaction.push_back(false);
    deque<PendingMemoryRequest> dpmr;
    pending_requests.push_back(dpmr);
  }

  conf_file.close();
}

void AnalyticalMemory::set_sys(int id, Sys* sys) {
  sys_map[id] = sys;
}

void AnalyticalMemory::issue(
    TensorLocationType tensor_loc,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
  int sys_id = wlhd->sys_id;

  if (tensor_loc == LOCAL_MEMORY) {
    uint64_t runtime = get_local_mem_runtime(tensor_size);
    Sys* sys = sys_map[sys_id];
    sys->register_event(
        wlhd->workload,
        EventType::General,
        wlhd,
        runtime);
  } else {
    if (mem_type == NO_MEMORY_EXPANSION) {
      cerr << "Remote memory access is not supported in NO_MEMORY_EXPANSION" << endl;
      exit(1);
    } else if (mem_type == PER_NODE_MEMORY_EXPANSION) {
      int nid = sys_id / num_npus_per_node;
      if (ongoing_transaction[nid]) {
        PendingMemoryRequest pmr(tensor_size, wlhd);
        pending_requests[nid].push_back(pmr);
      } else {
        uint64_t runtime = get_remote_mem_runtime(tensor_size);

        Sys* sys = sys_map[sys_id];
        sys->register_event(
            wlhd->workload,
            EventType::General,
            wlhd,
            runtime);

        sys->register_event(
            this,
            EventType::General,
            wlhd,
            runtime);

        ongoing_transaction[nid] = true;
      }
    } else if (mem_type == PER_NPU_MEMORY_EXPANSION) {
      uint64_t runtime = get_remote_mem_runtime(tensor_size);
      Sys* sys = sys_map[sys_id];
      sys->register_event(
          wlhd->workload,
          EventType::General,
          wlhd,
          runtime);
    } else if (mem_type == MEMORY_POOL) {
      if (ongoing_transaction[0]) {
        PendingMemoryRequest pmr(tensor_size, wlhd);
        pending_requests[0].push_back(pmr);
      } else {
        uint64_t runtime = get_remote_mem_runtime(tensor_size);

        Sys* sys = sys_map[sys_id];
        sys->register_event(
            wlhd->workload,
            EventType::General,
            wlhd,
            runtime);

        sys->register_event(
            this,
            EventType::General,
            wlhd,
            runtime);

        ongoing_transaction[0] = true;
      }
    }
  }
}

void AnalyticalMemory::call(EventType type, CallData* data) {
  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
    int nid = wlhd->sys_id / num_npus_per_node;
    if (!pending_requests[nid].empty()) {
      PendingMemoryRequest pmr = pending_requests[nid].front();
      pending_requests[nid].pop_front();

      uint64_t runtime = get_remote_mem_runtime(pmr.tensor_size);

      Sys* sys = sys_map[pmr.wlhd->sys_id];
      sys->register_event(
          pmr.wlhd->workload,
          EventType::General,
          pmr.wlhd,
          runtime);

      sys->register_event(
          this,
          EventType::General,
          pmr.wlhd,
          runtime);

      ongoing_transaction[nid] = true;
    } else {
      ongoing_transaction[nid] = false;
    }
  } else if (mem_type == MEMORY_POOL) {
    if (!pending_requests[0].empty()) {
      PendingMemoryRequest pmr = pending_requests[0].front();
      pending_requests[0].pop_front();

      uint64_t runtime = get_remote_mem_runtime(pmr.tensor_size);

      Sys* sys = sys_map[pmr.wlhd->sys_id];
      sys->register_event(
          pmr.wlhd->workload,
          EventType::General,
          pmr.wlhd,
          runtime);

      sys->register_event(
          this,
          EventType::General,
          pmr.wlhd,
          runtime);

      ongoing_transaction[0] = true;
    } else {
      ongoing_transaction[0] = false;
    }
  }
}

uint64_t AnalyticalMemory::get_local_mem_runtime(uint64_t tensor_size) {
  uint64_t runtime = local_mem_latency
    + static_cast<uint64_t>(
        (static_cast<double>(tensor_size) / local_mem_bw)
        * static_cast<double>(FREQ));
  return runtime;
}

uint64_t AnalyticalMemory::get_remote_mem_runtime(uint64_t tensor_size) {
  uint64_t runtime = remote_mem_latency
    + static_cast<uint64_t>(
        (static_cast<double>(tensor_size) / remote_mem_bw)
        * static_cast<double>(FREQ));
  return runtime;
}
