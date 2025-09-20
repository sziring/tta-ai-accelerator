# Synthesis Tool Assessment for TTA Research

**Date**: September 19, 2025
**Goal**: Evaluate synthesis toolchain options for credible ASIC implementation estimates

## 🔧 **Open-Source Synthesis Tools**

### **1. Yosys + OpenSTA + OpenROAD Flow**

**Pros:**
- ✅ **Completely Free**: No licensing barriers
- ✅ **Academic Friendly**: Widely used in research
- ✅ **Modern Capabilities**: Supports SystemVerilog, advanced optimizations
- ✅ **PDK Support**: Works with SkyWater 130nm, other open PDKs
- ✅ **Full Flow**: Logic synthesis → Place & Route → Timing analysis
- ✅ **Reproducible**: Consistent results across environments

**Cons:**
- ⚠️ **Limited Libraries**: Fewer standard cell options than commercial
- ⚠️ **Technology Nodes**: Primarily older processes (130nm, 180nm)
- ⚠️ **Tool Maturity**: Less optimized than decades-old commercial tools
- ⚠️ **Documentation**: Community-driven, sometimes sparse

**Best For**:
- Initial proof-of-concept synthesis
- Relative comparisons (TTA vs RISC)
- Academic publication where tool access is limited

### **2. OpenROAD-flow-scripts**

**Pros:**
- ✅ **Integrated Flow**: Automated synthesis-to-GDSII
- ✅ **Multiple PDKs**: SkyWater 130nm, TSMC (academic), others
- ✅ **Scripted Automation**: Easy to reproduce results
- ✅ **Active Development**: Google, DARPA funded project
- ✅ **Tutorial Support**: Good getting-started documentation

**Cons:**
- ⚠️ **Process Limitations**: Still mostly 130nm and older
- ⚠️ **Memory Compilers**: Limited SRAM generation options
- ⚠️ **Advanced Features**: Some commercial-tool features missing

**Assessment**: **RECOMMENDED FOR INITIAL STUDIES**

---

## 🏢 **Commercial Tool Access Options**

### **1. University Programs**

**Synopsys University Program:**
- Academic licenses for Design Compiler, IC Compiler, PrimeTime
- Access through university partnerships
- Requires institutional affiliation

**Cadence Academic Network:**
- Genus, Innovus, Tempus tools available
- University-hosted or cloud-based access
- Educational licensing restrictions

**Mentor Graphics (Siemens EDA):**
- Calibre, Tessent tools
- Academic partnerships available
- Focus on verification and DFT

### **2. Cloud-Based Access**

**EDA Playground:**
- Limited synthesis capabilities
- Good for small RTL validation
- Free tier available

**Cloud EDA Services:**
- AWS EDA instances with pre-installed tools
- Pay-per-use model
- Requires tool licenses

### **3. Evaluation Licenses**

**30-90 Day Trials:**
- Most vendors offer evaluation periods
- Full tool functionality
- Requires business justification

**Assessment**: **PURSUE UNIVERSITY ACCESS IF AVAILABLE**

---

## 📋 **Recommended Toolchain Strategy**

### **Phase 1: Open-Source Validation (Immediate)**
1. **Setup Yosys + OpenSTA + OpenROAD**
2. **Target SkyWater 130nm PDK**
3. **Validate synthesis methodology**
4. **Generate initial area/power estimates**

### **Phase 2: Commercial Tool Access (Parallel)**
1. **Research university partnerships**
2. **Contact EDA vendors for evaluation licenses**
3. **Target advanced nodes (28nm, 14nm)**
4. **Refine estimates with production tools**

### **Phase 3: Technology Scaling (Later)**
1. **Apply scaling models for advanced nodes**
2. **Use published scaling factors from literature**
3. **Validate against commercial tool results where available**

---

## 🎯 **Target Technology Nodes**

### **Open-Source Accessible:**
- **SkyWater 130nm**: Fully open PDK, good for relative comparisons
- **IHP 130nm**: Open BiCMOS process
- **GlobalFoundries 180nm**: Some open cell libraries

### **Commercial Targets:**
- **TSMC 28nm**: Mature node, good area/power balance
- **Samsung 14nm**: Advanced FinFET, relevant for mobile
- **Intel 7nm**: Industry-leading density

### **Scaling Strategy:**
- Use 130nm open-source for methodology validation
- Apply published scaling factors for advanced nodes
- Compare against literature results for validation

---

## 📊 **Expected Outputs**

### **Synthesis Reports Will Provide:**
1. **Area Breakdown**
   - Combinational logic area
   - Sequential logic area
   - Memory area
   - Total die area estimates

2. **Power Analysis**
   - Dynamic power consumption
   - Static (leakage) power
   - Clock power distribution
   - I/O power requirements

3. **Timing Analysis**
   - Critical path delays
   - Maximum clock frequency
   - Setup/hold slack margins
   - Clock skew analysis

4. **Resource Utilization**
   - Standard cell usage
   - Memory block requirements
   - I/O pin counts
   - Metal layer utilization

### **Comparative Metrics:**
- **TTA vs RISC**: Area efficiency, power efficiency
- **TTA vs GPU Tile**: Compute density, energy per operation
- **Technology Scaling**: Node-to-node improvements

---

## 🚀 **Immediate Action Items**

### **This Week (Sept 19-26)**
- [ ] **Install OpenROAD flow** on development machine
- [ ] **Download SkyWater 130nm PDK** and validate installation
- [ ] **Create simple test design** (basic ALU) to validate toolchain
- [ ] **Document installation process** for reproducibility

### **Next Week (Sept 26-Oct 3)**
- [ ] **Design VECMAC unit RTL** with realistic complexity
- [ ] **Synthesize first TTA component** and analyze results
- [ ] **Compare against simple RISC ALU** for methodology validation
- [ ] **Research university EDA access** options

### **Week 3 (Oct 3-10)**
- [ ] **Complete bus architecture synthesis**
- [ ] **Add memory hierarchy components**
- [ ] **Generate comprehensive area/power report**
- [ ] **Begin technology scaling analysis**

---

## 🔗 **Resources & Links**

### **Tool Documentation:**
- [OpenROAD Documentation](https://openroad.readthedocs.io/)
- [Yosys Manual](https://yosyshq.readthedocs.io/)
- [SkyWater PDK](https://skywater-pdk.readthedocs.io/)

### **Academic Papers:**
- "OpenROAD: An Open-Source Autonomous Physical Design Tool" (IEEE Design & Test)
- "A Fully Open Source Physical Design Ecosystem" (DAC 2019)
- "SkyWater Open Source PDK" (WOSET 2020)

### **Tutorial Resources:**
- OpenROAD Flow Scripts Tutorial
- SkyWater PDK Getting Started Guide
- Academic EDA tool access guides

**Next Update**: September 26, 2025