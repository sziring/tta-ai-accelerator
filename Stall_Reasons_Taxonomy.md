# TTA Simulator Stall Reasons Taxonomy

**Version**: v0.1  
**Last Updated**: 16/12/2024 16:15 UTC

## Stall Categories

### Resource Conflicts

#### `dst_busy`
**Description**: Destination functional unit is busy processing previous operation  
**Cause**: Attempted to write to FU that hasn't completed previous operation  
**Resolution**: Wait until FU.busy_until <= current_cycle  
**Energy Impact**: No transport energy consumed (move not executed)

#### `bus_full`
**Description**: All available buses are occupied this cycle  
**Cause**: More moves scheduled than bus capacity allows  
**Resolution**: Defer move to next cycle with available bus  
**Energy Impact**: No energy consumed (move not issued)

#### `port_conflict`
**Description**: Multiple moves target the same destination port in same cycle  
**Cause**: Scheduler error or WAW (Write-After-Write) hazard  
**Resolution**: Serialize moves or use different target ports  
**Energy Impact**: Only first move executes, others stalled

### Memory System Conflicts

#### `mem_conflict`
**Description**: Bank conflict in scratchpad memory system  
**Cause**: Multiple accesses to same SPM bank in same cycle  
**Details**: Bank selection based on address LSB (addr[0] for 2-bank system)  
**Resolution**: Queue conflicting access for next cycle  
**Energy Impact**: Successful access pays energy, stalled access pays none

#### `mem_latency`
**Description**: Memory operation still in progress  
**Cause**: Multi-cycle memory operation not yet complete  
**Resolution**: Wait for memory operation completion  
**Energy Impact**: Varies by memory type and operation

### Data Dependencies

#### `src_not_ready`
**Description**: Source port does not have valid data available  
**Cause**: Producer operation hasn't completed yet  
**Resolution**: Wait for producer to complete and update output port  
**Energy Impact**: No transport energy (move not executed)

#### `data_hazard`
**Description**: Read-After-Write dependency violation  
**Cause**: Consumer scheduled before producer completes  
**Resolution**: Insert NOP cycles or reschedule  
**Energy Impact**: No energy consumed for stalled move

### Scheduling Issues

#### `fu_latency`
**Description**: Functional unit has multi-cycle latency  
**Cause**: Operation inherently requires multiple cycles  
**Resolution**: Account for latency in scheduling  
**Energy Impact**: Full operation energy consumed, but output delayed

#### `pipeline_stall`
**Description**: Pipeline bubble due to control flow  
**Cause**: Predicated moves or conditional execution  
**Resolution**: Better branch prediction or speculative execution  
**Energy Impact**: Depends on speculation policy

## Stall Tracking Schema

### Per-Cycle Tracking
```csv
cycle,stall_type,count,moves_affected,energy_wasted
0,dst_busy,2,4,0.0
1,bus_full,1,1,0.0
2,mem_conflict,3,3,0.0
```

### Detailed Event Logging
```json
{
  "cycle": 5,
  "stalls": [
    {
      "type": "dst_busy",
      "src_port": "RF0.OUT",
      "dst_port": "ALU0.IN_A", 
      "fu_busy_until": 7,
      "energy_saved": 1.2
    },
    {
      "type": "mem_conflict",
      "address": "0x1004",
      "bank": 0,
      "conflicting_ops": ["read", "write"],
      "queue_depth": 2
    }
  ]
}
```

## Performance Impact Analysis

### Utilization Metrics
- **Bus Utilization**: `(successful_moves / (cycles * bus_count * issue_width)) * 100`
- **FU Utilization**: `(busy_cycles / total_cycles) * 100` per FU
- **Memory Utilization**: `(access_cycles / total_cycles) * 100` per bank

### Stall Impact Assessment
```rust
pub struct StallImpact {
    pub total_stalls: u64,
    pub stall_cycles: u64,
    pub energy_saved: f64,    // Energy not consumed due to stalls
    pub performance_loss: f64, // Percentage performance degradation
}
```

## Stall Prevention Strategies

### Scheduler Improvements
1. **Latency-Aware Scheduling**: Account for FU latencies in move ordering
2. **Resource-Aware Packing**: Consider bus and port capacity constraints
3. **Memory-Aware Placement**: Distribute memory accesses across banks

### Architecture Optimizations
1. **Additional Buses**: Reduce bus_full stalls
2. **Memory Banking**: Reduce mem_conflict stalls  
3. **FU Duplication**: Reduce dst_busy stalls for common operations

### Software Optimizations
1. **Loop Unrolling**: Reduce control flow stalls
2. **Data Layout**: Improve memory bank utilization
3. **Operation Fusion**: Combine dependent operations where possible

## Debugging Guidelines

### High dst_busy Rate (>20%)
- **Check**: FU latencies vs operation frequency
- **Action**: Add more FUs or pipeline operations
- **Target**: <10% for well-scheduled code

### High bus_full Rate (>15%)
- **Check**: Bus count vs issue width vs parallelism
- **Action**: Increase bus count or reduce issue width
- **Target**: <5% for balanced configurations

### High mem_conflict Rate (>25%)
- **Check**: Memory access patterns vs bank count
- **Action**: Increase bank count or improve data layout
- **Target**: <10% for well-distributed access patterns

## Stall Reason Priority

When multiple stall conditions occur simultaneously:

1. **dst_busy** (highest priority - FU resource)
2. **mem_conflict** (memory system resource)
3. **bus_full** (transport resource)  
4. **port_conflict** (logical conflict)
5. **src_not_ready** (data dependency)

Only the highest-priority stall reason is recorded per move.

## Energy Accounting for Stalls

### No Energy Consumed
- `dst_busy`, `bus_full`, `port_conflict`, `src_not_ready`
- Move is not issued, no transport occurs

### Partial Energy Consumed  
- `mem_conflict`: Queue management overhead (0.1 units)
- `pipeline_stall`: Speculative execution costs (varies)

### Full Energy Consumed
- `fu_latency`: Operation completes, just takes longer
- No stall energy impact (expected behavior)

## Validation Tests

### Stall Detection Accuracy
```rust
#[test]
fn test_dst_busy_detection() {
    // Write to ALU while it's busy
    // Verify stall_reason = dst_busy
    // Verify no energy consumed
}

#[test] 
fn test_mem_conflict_queueing() {
    // Two accesses to same bank same cycle
    // Verify one succeeds, one queued
    // Verify stall_reason = mem_conflict
}
```

### Stall Recovery Verification
```rust
#[test]
fn test_stall_recovery() {
    // Cause stall, then verify normal operation resumes
    // Check stall counters reset properly
    // Verify energy accounting remains consistent
}
```
