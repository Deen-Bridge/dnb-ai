"""Rights register, quarantine, and takedown behaviour for bundled hadith data."""

import json
import shutil

import hadith
from hadith import BundledGradingSource, annotate


def _copy_data(tmp_path):
    dest = tmp_path / "hadith"
    shutil.copytree(hadith.DATA_DIR, dest)
    return dest


def _set_status(data_dir, collection, status):
    path = data_dir / hadith.RIGHTS_REGISTER_NAME
    register = json.loads(path.read_text(encoding="utf-8"))
    register["collections"][collection]["status"] = status
    path.write_text(json.dumps(register), encoding="utf-8")


def test_every_bundled_collection_is_registered_with_full_metadata():
    for path in hadith.DATA_DIR.glob("*.json"):
        if path.name in ("variants.json", hadith.RIGHTS_REGISTER_NAME):
            continue
        rights = hadith.collection_rights(path.stem)
        assert rights is not None, f"{path.stem} missing from rights register"
        for key in ("origin", "version", "license", "permission", "attribution", "retrieved", "status"):
            assert rights.get(key), f"{path.stem} missing {key}"


def test_citation_identifies_authorized_source():
    refs = annotate("Sahih al-Bukhari 1 is narrated.")
    assert refs and refs[0].source
    assert "fawazahmed0/hadith-api" in refs[0].source
    assert "Unlicense" in refs[0].source


def test_quarantined_collection_is_not_served(tmp_path):
    data_dir = _copy_data(tmp_path)
    _set_status(data_dir, "bukhari", "quarantined")
    source = BundledGradingSource(data_dir)
    assert source.get("bukhari", 1) is None
    assert source.get("muslim", 1) is not None
    assert BundledGradingSource(data_dir, allow_unverified=True).get("bukhari", 1) is not None


def test_takedown_withdraws_collection(tmp_path):
    data_dir = _copy_data(tmp_path)
    _set_status(data_dir, "muslim", "withdrawn")
    assert not hadith.is_released("muslim", data_dir)
    assert hadith.source_citation("muslim", data_dir) is None
    assert BundledGradingSource(data_dir).get("muslim", 1) is None


def test_unregistered_or_missing_register_fails_closed(tmp_path):
    data_dir = _copy_data(tmp_path)
    (data_dir / hadith.RIGHTS_REGISTER_NAME).unlink()
    assert BundledGradingSource(data_dir).get("bukhari", 1) is None
