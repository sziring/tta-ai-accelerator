# Functional Unit Port Contracts

**Version**: v0.1  
**Last Updated**: 16/12/2024 16:10 UTC

## Port Naming Standard

All functional units follow this consistent naming pattern:

### Input Ports
- **IN_A, IN_B, IN_C**: Primary operand inputs
- **ACC_IN**: Accumulator input (for MAC operations)
- **ADDR_IN**: Address input (for memory operations)
- **DATA_IN**: Data input (for writes)
- **VEC_IN**: Vector input
- **MODE_IN**: Operation mode selection
- **TRIG_IN**: Generic trigger port (when operation is mode-dependent)

### Output Ports
- **OUT**: Primary scalar output
- **DATA_OUT**: Data output (for reads)
- **SCALAR_OUT**: Scalar result from vector operations
- **VEC_OUT**: Vector output
- **READY_OUT**: Ready signal output

### Trigger Port Rules
Each FU specifies exactly which port(s) trigger operation execution:
- Writing to a trigger port initiates the operation
- Non-trigger ports are latched inputs that persist until overwritten
- Operations only execute when trigger port is written AND all required inputs are available

## Functional Unit Specifications

### ALU (Arithmetic Logic Unit)
```
Inputs:
  0: IN_A      (operand A, latched)
  1: IN_B      (operand B, TRIGGER)

Outputs:
  0: OUT       (result)

Operation: OUT = IN_A + IN_B
Trigger: Write to IN_B executes addition
Latency: 1 cycle
Energy: 8 units (add16)
```

### RegisterFile
```
Inputs:
  0: READ_ADDR   (read address, immediate effect)
  1: WRITE_ADDR  (write address, latched)
  2: WRITE_DATA  (write data, TRIGGER)

Outputs:
  0: READ_OUT    (read result)

Operations:
  - Write to READ_ADDR: READ_OUT = RF[READ_ADDR] (immediate)
  - Write to WRITE_DATA: RF[WRITE_ADDR] = WRITE_DATA (trigger)
Latency: 1 cycle (combinational read, registered write)
Energy: 4 units (read), 6 units (write)
```

### VECMAC (Vector Multiply-Accumulate)
```
Inputs:
  0: VEC_A     (vector A, latched)
  1: VEC_B     (vector B, TRIGGER)
  2: ACC_IN    (accumulator input, optional)
  3: MODE_IN   (accumulate=1/clear=0, latched)

Outputs:
  0: SCALAR_OUT (scalar accumulation result)

Operation: 
  if MODE_IN: SCALAR_OUT = ACC_IN + sum(VEC_A[i] * VEC_B[i])
  else:       SCALAR_OUT = sum(VEC_A[i] * VEC_B[i])
Trigger: Write to VEC_B executes MAC
Latency: 2-3 cycles (depends on lane count)
Lanes: 8 or 16 (configurable)
Energy: 40 units (vecmac8x8_to_i32 for 16 lanes)
```

### REDUCE (Vector Reduction)
```
Inputs:
  0: VEC_IN    (vector input, TRIGGER)
  1: MODE_IN   (0=sum, 1=max, 2=argmax, latched)

Outputs:
  0: SCALAR_OUT (reduction result)
  1: INDEX_OUT  (index for argmax, valid only for mode=2)

Operations:
  - MODE_IN=0: sum all elements
  - MODE_IN=1: find maximum value
  - MODE_IN=2: find index of maximum value
Trigger: Write to VEC_IN executes reduction
Latency: 1 cycle (sum/max), 2 cycles (argmax)
Energy: 10 units (sum), 16 units (argmax)
```

### SPM (Scratchpad Memory)
```
Inputs:
  0: ADDR_IN     (address, latched)
  1: DATA_IN     (write data, latched)
  2: READ_TRIG   (triggers read operation)
  3: WRITE_TRIG  (triggers write operation)

Outputs:
  0: DATA_OUT    (read result)

Operations:
  - Write to READ_TRIG: DATA_OUT = SPM[ADDR_IN]
  - Write to WRITE_TRIG: SPM[ADDR_IN] = DATA_IN
Configuration:
  - Banks: 2 (configurable)
  - Size: 16KiB per bank
  - Bank selection: ADDR_IN[0] (even/odd)
Latency: 1 cycle
Energy: 10 units (read), 12 units (write)
Conflicts: Bank conflicts cause stalls (stall_reason=mem_conflict)
```

### IMM (Immediate Value Unit)
```
Inputs:
  0: SELECT_IN  (constant index, immediate effect)

Outputs:
  0: OUT        (selected constant)

Operation: OUT = constants[SELECT_IN]
Configuration: Pre-loaded constant table
Latency: 0 cycles (combinational)
Energy: 0 units (no computation)
```

## Port Address Mapping

Each functional unit is assigned a FU ID, and ports are numbered within each FU:

```rust
// Example port addresses
ALU0.IN_A     = PortId { fu: 0, port: 0 }
ALU0.IN_B     = PortId { fu: 0, port: 1 }
ALU0.OUT      = PortId { fu: 0, port: 0 }

RF0.READ_ADDR = PortId { fu: 1, port: 0 }
RF0.WRITE_DATA= PortId { fu: 1, port: 2 }
RF0.READ_OUT  = PortId { fu: 1, port: 0 }
```

## Error Conditions

### Invalid Port Access
- Reading from non-existent port: Returns None
- Writing to non-existent port: Returns FuEvent::Error

### Resource Conflicts
- Writing to busy FU: Returns FuEvent::BusyUntil(cycle)
- Bank conflicts in SPM: Queued with stall tracking

### Data Type Mismatches
- Wrong data type for port: Returns FuEvent::Error with description
- Vector length mismatch: Truncate or pad with warnings

## Timing Model

### Pipeline Stages
1. **Transport**: Data moves on buses (energy cost)
2. **Write**: Data written to input ports (may trigger)
3. **Execute**: FU performs operation (latency cycles)
4. **Ready**: Result available on output ports

### Timing Guarantees
- Results available exactly at `current_cycle + latency`
- Trigger ports accept new operations only when FU is idle
- Read ports provide stable data until next write to same FU

## Energy Accounting

### Per-Operation Costs
All energy costs in abstract units (calibrated to physics engine):

```toml
[fu_energy]
add16 = 8
sub16 = 8
mul16 = 24
vecmac8x8_to_i32 = 40
reduce_sum16 = 10
reduce_argmax16 = 16
regfile_read = 4
regfile_write = 6
spm_read_32b = 10
spm_write_32b = 12
```

### Transport Costs
```toml
[bus_energy]
alpha_per_bit_toggle = 0.02
beta_base = 1.0
```

## Validation Requirements

### Unit Tests
Each FU must pass:
1. **Correctness**: Golden reference validation
2. **Timing**: Output ready at exact cycle
3. **Energy**: Cost matches table ±0.5 units
4. **Trigger Semantics**: No-op unless trigger written

### Property Tests
- Commutativity where applicable (ADD, MUL)
- Associativity for reductions
- Data type preservation through pipelines

### Integration Tests
- Multi-FU operations (ALU → RF → SPM)
- Resource conflict handling
- Energy accounting across operations
