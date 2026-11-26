# Software validation status

The release package was checked locally without running CARLA or training any model.

- Python compilation: all included Python modules passed.
- Shell syntax: all included shell scripts passed `bash -n`.
- USPI deterministic CPU unit tests: 10/10 passed (PyTorch 2.10.0+cpu, NumPy 2.3.5 in the packaging environment, not the historical CUDA runtime).
- Extended evaluator argument parser: `python evaluate_uspi.py --help` passed.
- Historical dashboard comparison adapter: empty-data JSON serialization passed; no invented evaluation values.
- Dashboard JavaScript syntax: Node.js syntax check passed.
- Archive: ZIP integrity test passed.

Not verified: complete upstream clone and dependency installation on a new cloud server, actual historical weights loading (weights not supplied), CARLA integration of USPI, live camera performance or browser end-to-end video, runtime inference latency of USPI, empirical policy performance, mathematical stability/safety guarantees. No new USPI driving evaluation or training was performed. The repository is a proposed research implementation, not an experimentally validated release.
