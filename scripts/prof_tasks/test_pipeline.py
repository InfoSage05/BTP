"""Guards for the CHF pipeline. Each test pins a bug that was actually found."""
import numpy as np
import pytest

import pipeline as PL
import run_pipeline as RP


@pytest.fixture(scope="module")
def fitted():
    return PL.CHFPipeline().fit(RP.build_corpus())


def test_rejects_pool_boiling(fitted):
    # G=0 silently returned 0.0 before validation existed
    with pytest.raises(ValueError, match="mass flux must be positive"):
        fitted.predict_one(fluid="water", P_kPa=101, G_kg_m2s=0, D_mm=8,
                           L_mm=1000, Tin_C=90)


def test_rejects_bad_geometry(fitted):
    with pytest.raises(ValueError, match="must be positive"):
        fitted.predict_one(fluid="water", P_kPa=10000, G_kg_m2s=2000, D_mm=-8,
                           L_mm=1000, Tin_C=200)


def test_rejects_supercritical(fitted):
    with pytest.raises(ValueError, match="critical pressure"):
        fitted.predict_one(fluid="water", P_kPa=25000, G_kg_m2s=2000, D_mm=8,
                           L_mm=1000, Tin_C=300)


def test_rejects_unknown_fluid(fitted):
    with pytest.raises(ValueError, match="unknown fluid"):
        fitted.predict_one(fluid="unobtainium", P_kPa=1000, G_kg_m2s=500,
                           D_mm=8, L_mm=1000, Tin_C=20)


def test_unseen_fluid_uses_its_own_properties(fitted):
    """The original bug: any unrecognised fluid silently got water's properties."""
    w = fitted.predict_one(fluid="water", P_kPa=1000, G_kg_m2s=500, D_mm=8,
                           L_mm=1000, Tin_C=20)
    a = fitted.predict_one(fluid="ammonia", P_kPa=1000, G_kg_m2s=500, D_mm=8,
                           L_mm=1000, Tin_C=20)
    assert a["CHF_kW_m2"] != w["CHF_kW_m2"]
    assert any("no training data" in m for m in a["warnings"])


def test_interval_brackets_the_point(fitted):
    r = fitted.predict_one(fluid="water", P_kPa=10000, G_kg_m2s=2000, D_mm=8,
                           L_mm=1000, Tin_C=200)
    lo, hi = r["interval_kW_m2"]
    assert lo < r["CHF_kW_m2"] < hi and lo > 0


def test_never_silently_returns_zero(fitted):
    r = fitted.predict_one(fluid="water", P_kPa=10000, G_kg_m2s=2000, D_mm=8,
                           L_mm=1000, Tin_C=200)
    assert np.isfinite(r["CHF_kW_m2"]) and r["CHF_kW_m2"] > 0


def test_blend_free_stays_off():
    """Blending toward the unanchored model took held-out D7 to R2 -206.9."""
    assert PL.CHFPipeline().blend_free is False


def test_roundtrip(tmp_path, fitted):
    p = tmp_path / "pipe.pkl"
    fitted.save(str(p))
    back = PL.CHFPipeline.load(str(p))
    kw = dict(fluid="water", P_kPa=10000, G_kg_m2s=2000, D_mm=8, L_mm=1000, Tin_C=200)
    assert back.predict_one(**kw)["CHF_kW_m2"] == fitted.predict_one(**kw)["CHF_kW_m2"]
