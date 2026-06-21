from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_WORKFLOW = ROOT / ".github" / "workflows" / "bench.yml"


def _workflow_block(workflow: str, start: str, end: str) -> str:
    start_idx = workflow.index(start)
    end_idx = workflow.index(end, start_idx)
    return workflow[start_idx:end_idx]


def test_benchmark_workflow_enforces_pr_smoke_test_before_comparison():
    workflow = BENCH_WORKFLOW.read_text()

    smoke_block = _workflow_block(workflow, "Smoke test benchmarks (PR)", "Run benchmarks (PR)")

    assert "run: cargo bench -- --test" in smoke_block
    assert "continue-on-error: true" not in smoke_block
    assert "|| true" not in smoke_block


def test_benchmark_workflow_keeps_full_comparison_nonblocking():
    workflow = BENCH_WORKFLOW.read_text()

    pr_block = _workflow_block(workflow, "Run benchmarks (PR)", "Checkout base branch")
    base_block = _workflow_block(workflow, "Run benchmarks (base)", "Compare benchmarks")

    assert "continue-on-error: true" in pr_block
    assert "cargo bench 2>&1 | tee pr-bench.txt || true" in pr_block
    assert "continue-on-error: true" in base_block
    assert "cargo bench 2>&1 | tee base-bench.txt || true" in base_block
