src/
├── main.rs              # CLI interface
├── lib.rs               # Library exports
├── config/
│   └── mod.rs           # Configuration system
├── physics/
│   ├── mod.rs           # Physics module exports
│   ├── universe.rs      # Core physics engine
│   └── energy_validation.rs # Validation interface
├── tta/
│   ├── mod.rs           # TTA core exports
│   ├── processor.rs     # Your existing processor
│   ├── functional_unit.rs  # FU trait definition (FILE, not directory)
│   ├── immediate_unit.rs   # IMM implementation  
│   └── spm_unit.rs        # SPM implementation
└── tests/
    └── integration_test.rs # Integration tests
