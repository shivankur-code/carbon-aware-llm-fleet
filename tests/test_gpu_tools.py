"""CPU tests for the GPU tooling (fault-injection logic, ingest round-trip)."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import CONFIGS, ROOT

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "gpu"))
from c3_kv_faults import inject  # noqa: E402

from carbonsched.hardware import load_catalog  # noqa: E402


def kv(seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 2, 64, 16, generator=g).to(dtype) for _ in range(4)]


def n_diff(a, b):
    return sum(int((x.view(torch.int16) != y.view(torch.int16)).sum()) for x, y in zip(a, b))


def test_zero_ber_changes_nothing():
    a = kv()
    b = [t.clone() for t in a]
    s = inject(b, 0.0, "none", np.random.default_rng(0))
    assert s["flips"] == 0 and n_diff(a, b) == 0


def test_unprotected_flips_land_and_checksum_hides_all():
    a = kv()
    b = [t.clone() for t in a]
    s = inject(b, 1e-3, "none", np.random.default_rng(1))
    assert s["flips"] > 0 and s["silent"] == s["flips"]
    assert 0 < n_diff(a, b) <= s["flips"]
    c = [t.clone() for t in a]
    s = inject(c, 1e-3, "checksum", np.random.default_rng(1))
    assert s["silent"] == 0 and n_diff(a, c) == 0 and s["recomputed_blocks"] > 0


def test_secded_corrects_isolated_flips():
    a = kv()
    b = [t.clone() for t in a]
    s = inject(b, 1e-5, "secded", np.random.default_rng(2))   # flips almost surely in distinct words
    assert s["corrected"] == s["flips"] and n_diff(a, b) == 0


def test_bit_class_restricts_positions():
    a = kv(dtype=torch.bfloat16)
    b = [t.clone() for t in a]
    inject(b, 1e-2, "none", np.random.default_rng(3), bit_class="mantissa")
    for x, y in zip(a, b):
        xor = (x.view(torch.int16) ^ y.view(torch.int16)).to(torch.int32) & 0xFFFF
        assert int((xor & ~0x7F).sum()) == 0   # only bits 0..6 (bf16 mantissa) changed


def test_ingest_roundtrip(tmp_path):
    mk = lambda lc, bt, idle, busy: dict(tp=1, summary=dict(  # noqa: E731
        lc_out_tok_per_s=lc, batch_out_tok_per_s=bt, idle_power_w=idle, batch_mean_power_w=busy, lc_mean_power_w=busy))
    (tmp_path / "old.json").write_text(json.dumps(mk(300, 1500, 60, 380)))
    (tmp_path / "new.json").write_text(json.dumps(mk(1000, 3000, 90, 650)))
    cfg = tmp_path / "configs"
    shutil.copytree(CONFIGS, cfg)
    subprocess.run([sys.executable, str(ROOT / "gpu" / "ingest.py"), "--c2-old", str(tmp_path / "old.json"),
                    "--c2-new", str(tmp_path / "new.json"), "--out", str(cfg / "measured.yaml")], check=True)
    cat = load_catalog(cfg)
    assert cat.skus["old_gpu"].throughput == {"LC": pytest.approx(0.3), "BATCH": pytest.approx(0.5)}
    assert cat.registry["skus.skus.old_gpu.throughput.BATCH"].status == "measured"
    assert cat.skus["new_gpu"].power_peak_w == 8 * 650 + 2000
    # green_gpu power follows new_gpu (power_relative_to)
    assert cat.skus["green_gpu"].power_peak_w == cat.skus["new_gpu"].power_peak_w - 100
    assert load_catalog(cfg, use_measured=False).skus["old_gpu"].throughput["BATCH"] == pytest.approx(0.45)
