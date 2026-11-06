## 📌 Pull Request Overview

### 1. Feature / Subtask Identification
- **Phase / Milestone:** <!-- e.g., Phase 3: Advanced Traffic Engineering -->
- **Subtask ID:** <!-- e.g., 3.1 (IPM Multi-Approach Calibration), 3.2 (Emergency Preemption), 3.3 (Municipal Vector PDF Report) -->
- **Branch Name:** `<!-- e.g., feature/3.1-ipm-calibration -->`
- **Target Branch:** `main`

---

### 2. Type of Change
- [ ] `feat`: A new feature or analytical capability
- [ ] `fix`: A bug fix or error resolution
- [ ] `refactor`: Code reorganization with identical functionality
- [ ] `test`: Added or updated unit/integration tests
- [ ] `perf`: Algorithmic or simulation performance optimization
- [ ] `docs`: Documentation, guides, or report schema updates
- [ ] `ci`: CI/CD pipeline or developer workflow updates

---

### 3. Summary of Changes
<!-- Provide a concise 2-4 sentence explanation of the technical changes made in this branch. -->

---

### 4. Impacted Modules & Files
- [ ] Vision & Kinematics (`src/analysis/`, `src/inference/`)
- [ ] Reinforcement Learning & Actuation (`src/rl/`, `src/decision/`)
- [ ] Eclipse SUMO Digital Twin (`src/sumo/`)
- [ ] Database & Telemetry (`src/db/`)
- [ ] Municipal Reports & Exports (`src/reports/`)
- [ ] Streamlit Web UI (`app/`)
- [ ] Unit Tests (`tests/`)

---

### 5. Verification & Test Evidence
- **Local Test Suite:** `115 / 115 tests passed`
- **Flake8 Syntax Check:** `0 errors`
- **Simulation Verification:**
  <!-- Note whether SUMO headless or Streamlit UI was manually verified -->
  - [x] Streamlit UI runs cleanly without runtime exception
  - [x] Pytest suite passes locally: `pytest tests/ -v`
  - [x] CI Pipeline passed on GitHub Actions

---

### 6. Production Quality Checklist
- [x] Dedicated feature branch created from `main` (never direct commit to `main`).
- [x] Atomic conventional commit messages used (`feat(...)`, `fix(...)`, etc.).
- [x] Zero hardcoded secrets, API keys, or proprietary cloud dependencies (100% free/local stack).
- [x] All existing regression tests preserved and passing.
