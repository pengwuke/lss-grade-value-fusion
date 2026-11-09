"""Manifest-bound access to exported tensor-only checkpoints."""

import hashlib
import json
from pathlib import Path
import re

from .contracts import fold_index


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _bound_json(models_root, filename, fingerprint_key):
    """Check package metadata consistency before parsing the same raw bytes.

    The adjacent compatibility record is not an external signature or trust
    anchor: editing both metadata and that record can produce a new matching
    pair. This catches accidental edits and mixed files within a model package.
    """
    root = Path(models_root).expanduser().resolve()
    try:
        record = json.loads((root / "CODE_MODEL_COMPATIBILITY.json").read_bytes())
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("Missing or invalid CODE_MODEL_COMPATIBILITY.json") from error
    expected = record.get(fingerprint_key) if isinstance(record, dict) else None
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
        raise ValueError("Missing or invalid SHA256 field: " + fingerprint_key)
    try:
        raw = (root / filename).read_bytes()
    except OSError as error:
        raise ValueError("Cannot read bound model metadata: " + filename) from error
    if hashlib.sha256(raw).hexdigest() != expected.lower():
        raise ValueError(filename + " SHA256 differs from CODE_MODEL_COMPATIBILITY.json")
    try:
        return json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise ValueError("Invalid JSON in bound model metadata: " + filename) from error


def read_manifest(models_root):
    manifest = _bound_json(models_root, "models_manifest.json", "model_manifest_sha256")
    entries = manifest["model_entries"]
    identities = [(e["group"], e["family"], fold_index(e["fold"])) for e in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate model role in manifest")
    return entries


def checkpoint_path(models_root, entry, verify=True):
    root = Path(models_root).expanduser().resolve()
    relative = Path(entry["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Checkpoint path must remain inside the model package")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint path escaped the model package")
    if path.stat().st_size != entry["bytes"]:
        raise ValueError("Checkpoint size differs from manifest")
    if verify and sha256(path) != entry["sha256"]:
        raise ValueError("Checkpoint SHA256 differs from manifest")
    return path


def find_entry(models_root, group, family, fold):
    fold = fold_index(fold)
    matches = [e for e in read_manifest(models_root)
               if (e["group"], e["family"], e["fold"]) == (group, family, fold)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one manifest-bound model role")
    return matches[0]


def read_fusion_config(models_root):
    return _bound_json(models_root, "fusion_config.json", "fusion_config_sha256")
