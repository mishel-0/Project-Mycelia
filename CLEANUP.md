# Clean build migration

The prior Structured/OLD implementations, classifier SDK, cached bytecode, historical simulation outputs and old ZIPs were removed from the active project after a verified recovery archive was created.

Existing `.venv` and MRI/OMNIA datasets remain on the Desktop. They are not required by this simulator and are not included in the clean source ZIP.

Recovery archive: `/Users/misheladnan/Documents/Codex/2026-10-02/https-chatgpt-com-share-6abf7b15-9ad0/outputs/MYCELIA-original-recovery.zip`. Extract its `Project-Mycelia-original` folder into a separate directory to recover the previous source and results. `RECOVERY_MANIFEST.json` records each original file and SHA-256.

This experimental simulator uses dimensionless model units and uncalibrated parameters. It supports compartment physiology, conservative resource transport, branching, fusion and remodeling; it does not claim molecular or species-specific fidelity.
