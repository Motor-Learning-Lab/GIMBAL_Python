# ⚠️ IMPORTANT: File Naming and Organization Conventions

**Read this before creating any diagnostic tests or reports.**

This document establishes mandatory naming conventions for all diagnostic work in GIMBAL. Following these conventions ensures:
- Tests are easy to find and understand
- Results are traceable and reproducible
- Reports clearly link to their corresponding tests
- The codebase remains organized as complexity grows

---

## Core Principle: One Test Plan → One Diagnostic Directory

Each test plan document in `plans/` should have a corresponding directory in `tests/diagnostics/` with a matching name structure.

**Example:**
- Plan: `plans/v0.2.1_divergence_test_plan.md`
- Directory: `tests/diagnostics/v0_2_1_divergence/`

Within each diagnostic directory, use the three-file pattern for each test:
1. `test_*.py` - Executable test code
2. `results_*.json` - Raw numerical output
3. `report_*.md` - Human-readable analysis

---

## Naming Patterns

### Test Scripts (Python)

**Pattern:** `test_group_N_<short_description>.py`

**Rules:**
- Must start with `test_` prefix (makes them identifiable as test files)
- Include `_group_N_` where N is the test group number from the plan
- Use lowercase with underscores for the description
- Description should be 2-4 words maximum

**Examples:**
```
test_group_1_baseline_no_hmm.py
test_group_2_baseline_with_hmm.py
test_group_3_state_count.py
test_group_5_divergence_localization.py
```

**What to include:**
- Docstring at top explaining purpose and linking to test plan
- Configuration parameters clearly defined
- Function(s) to run the test and collect metrics
- Code to save results to corresponding JSON file
- Return metrics for interactive use

---

### Results Files (JSON)

**Pattern:** `results_group_N_<short_description>.json`

**Rules:**
- Must start with `results_` prefix
- Use same `_group_N_` and description as corresponding test file
- JSON format for structured data
- Include metadata (timestamp, configuration, environment)

**Examples:**
```
results_group_1_baseline_no_hmm.json
results_group_2_baseline_with_hmm.json
```

**Required fields:**
```json
{
  "test_group": 1,
  "description": "Short description matching filename",
  "configuration": {
    "parameter1": value1,
    "parameter2": value2
  },
  "metrics": {
    "primary_metric": value,
    "secondary_metrics": {}
  },
  "timestamp": "ISO 8601 format",
  "environment": {
    "relevant": "version info"
  }
}
```

---

### Report Files (Markdown)

**Pattern:** `report_group_N_<short_description>.md`

**Rules:**
- Must start with `report_` prefix
- Use same `_group_N_` and description as corresponding test file
- Markdown format for readability
- Include interpretation and recommendations

**Examples:**
```
report_group_1_baseline_no_hmm.md
report_group_2_baseline_with_hmm.md
report_summary.md  (special case: synthesizes all tests)
```

**Required sections:**
1. **Test Header** - Group number, description, configuration, date
2. **Results Summary** - Key metrics in table format
3. **Interpretation** - What the results mean
4. **Diagnostic Artifacts** - Links to plots, traces, etc.
5. **Recommendations** - Next steps based on findings

---

### Diagnostic Plots (Directories)

**Pattern:** `plots/group_N_<short_description>/`

**Rules:**
- Subdirectory named `plots/` within the diagnostic directory
- Each test group gets its own subdirectory
- Use same `_group_N_` and description as test files
- Contains PNG/PDF files with descriptive names

**Example structure:**
```
tests/diagnostics/v0_2_1_divergence/
├── plots/
│   ├── group_1_baseline_no_hmm/
│   │   ├── divergence_pairs.png
│   │   ├── trace_plot.png
│   │   └── energy_plot.png
│   ├── group_2_baseline_with_hmm/
│   │   └── ...
```

---

## Example: Complete Test Group

For Test Group 1 from `v0.2.1_divergence_test_plan.md`:

```
tests/diagnostics/v0_2_1_divergence/
├── test_group_1_baseline_no_hmm.py      ← Runs the test
├── results_group_1_baseline_no_hmm.json ← Stores metrics
├── report_group_1_baseline_no_hmm.md    ← Analyzes results
└── plots/
    └── group_1_baseline_no_hmm/
        ├── divergence_pairs.png
        └── ess_comparison.png
```

**Workflow:**
1. Write `test_group_1_baseline_no_hmm.py` following the plan
2. Run it: `pixi run python test_group_1_baseline_no_hmm.py`
3. It generates `results_group_1_baseline_no_hmm.json` automatically
4. It generates plots in `plots/group_1_baseline_no_hmm/`
5. Write `report_group_1_baseline_no_hmm.md` interpreting the results
6. Update `report_summary.md` with findings from this test

---

## Special Files

### Summary Reports

**Name:** `report_summary.md` (in the diagnostic directory root)

**Purpose:** Synthesizes findings across all test groups

**Contents:**
- Table comparing key metrics across all tests
- Overall interpretation of what causes the issue
- Prioritized recommendations
- Links to individual group reports for details

### Utility Scripts

**Pattern:** `utils_<purpose>.py` or `<purpose>_utils.py`

**Examples:**
```
test_utils.py          (shared test utilities)
plotting_utils.py      (visualization helpers)
analysis_utils.py      (metric computation)
```

**Rules:**
- Do NOT start with `test_` or `results_` or `report_`
- Use descriptive names indicating their utility purpose
- Can be imported by multiple test scripts

---

## Benefits of This System

**Clarity:**
- File purpose is immediately obvious from name
- Test group numbers link related files together
- No confusion between tests, results, and reports

**Traceability:**
- Each result file traces to exactly one test script
- Each report references specific result files
- Summary report synthesizes everything

**Scalability:**
- Easy to add new test groups without naming conflicts
- Multiple test plans can coexist in separate directories
- Consistent structure across all diagnostics

**Maintainability:**
- Future developers understand structure instantly
- AI assistants can navigate the organization
- Documentation naturally stays aligned with code

---

## Anti-Patterns to Avoid

❌ **Don't do this:**
```
test1.py                    (no description)
baseline_test.py            (no group number)
hmm_test.py                 (no group number)
output.json                 (which test?)
results.json                (which test?)
analysis.md                 (which test?)
report.md                   (which test?)
```

✅ **Do this instead:**
```
test_group_1_baseline_no_hmm.py
test_group_2_baseline_with_hmm.py
results_group_1_baseline_no_hmm.json
results_group_2_baseline_with_hmm.json
report_group_1_baseline_no_hmm.md
report_group_2_baseline_with_hmm.md
report_summary.md
```

---

## Checklist for Creating New Diagnostics

Before creating files for a new test:

- [ ] Read the test plan document in `plans/`
- [ ] Identify the test group number
- [ ] Choose a short, descriptive name (2-4 words)
- [ ] Create test script: `test_group_N_<description>.py`
- [ ] Ensure it saves to: `results_group_N_<description>.json`
- [ ] Create plot directory: `plots/group_N_<description>/`
- [ ] Write report: `report_group_N_<description>.md`
- [ ] Update: `report_summary.md` with new findings
- [ ] Verify all files have matching `_group_N_<description>` suffixes

---

## Why This File Has This Name

This file is named `IMPORTANT_FILE_NAMING_CONVENTIONS.md` with "IMPORTANT" in all caps at the root level to ensure:

1. **It appears at the top of directory listings** (capitals sort before lowercase)
2. **The filename itself communicates urgency** (IMPORTANT grabs attention)
3. **It's in the root directory** (impossible to miss)
4. **It's explicit about content** (NAMING_CONVENTIONS tells you what's inside)

When working on GIMBAL diagnostics, this file should be your first reference for how to organize your work.

---

**Last updated:** December 10, 2025
**Applies to:** All diagnostic testing in GIMBAL v0.2.1 and beyond
