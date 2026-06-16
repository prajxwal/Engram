"""Tests for EngramConfig — layered configuration loading."""

import json
import os

import pytest

from engram.config import EngramConfig


class TestDefaults:
    """Tests for default values."""

    def test_default_values(self):
        config = EngramConfig()
        assert config.decay_rate == 0.005
        assert config.top_k == 10
        assert config.top_n == 5
        assert config.context_budget_ratio == 0.25
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.default_model == "llama3"

    def test_derived_paths(self):
        config = EngramConfig(data_dir="/tmp/test_engram")
        assert config.db_path.endswith("engram.db")
        assert "chroma" in config.chroma_path
        assert config.config_file_path.endswith("config.json")


class TestConfigLayering:
    """Tests for config priority: overrides > env > file > defaults."""

    def test_file_overrides_defaults(self, tmp_path):
        data_dir = str(tmp_path / "engram")
        os.makedirs(data_dir, exist_ok=True)

        config_file = os.path.join(data_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump({"decay_rate": 0.01, "top_k": 20}, f)

        config = EngramConfig.load(data_dir=data_dir)
        assert config.decay_rate == 0.01
        assert config.top_k == 20

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        data_dir = str(tmp_path / "engram")
        os.makedirs(data_dir, exist_ok=True)

        config_file = os.path.join(data_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump({"top_k": 20}, f)

        monkeypatch.setenv("ENGRAM_TOP_K", "30")
        config = EngramConfig.load(data_dir=data_dir)
        assert config.top_k == 30

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_TOP_K", "30")
        config = EngramConfig.load(top_k=50)
        assert config.top_k == 50

    def test_bad_env_var_ignored(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_TOP_K", "not_a_number")
        config = EngramConfig.load()
        assert config.top_k == 10  # Default

    def test_corrupted_config_file_ignored(self, tmp_path):
        data_dir = str(tmp_path / "engram")
        os.makedirs(data_dir, exist_ok=True)

        config_file = os.path.join(data_dir, "config.json")
        with open(config_file, "w") as f:
            f.write("not valid json{{{")

        config = EngramConfig.load(data_dir=data_dir)
        assert config.decay_rate == 0.005  # Falls back to default


class TestSetValue:
    """Tests for CLI-style set_value."""

    def test_set_float(self, tmp_path):
        config = EngramConfig(data_dir=str(tmp_path / "engram"))
        config.ensure_dirs()
        config.set_value("decay_rate", "0.01")
        assert config.decay_rate == 0.01

    def test_set_int(self, tmp_path):
        config = EngramConfig(data_dir=str(tmp_path / "engram"))
        config.ensure_dirs()
        config.set_value("top_k", "20")
        assert config.top_k == 20

    def test_set_string(self, tmp_path):
        config = EngramConfig(data_dir=str(tmp_path / "engram"))
        config.ensure_dirs()
        config.set_value("default_model", "mistral")
        assert config.default_model == "mistral"

    def test_set_invalid_key_raises(self, tmp_path):
        config = EngramConfig(data_dir=str(tmp_path / "engram"))
        with pytest.raises(ValueError, match="Unknown config key"):
            config.set_value("nonexistent_key", "value")

    def test_set_persists_to_file(self, tmp_path):
        config = EngramConfig(data_dir=str(tmp_path / "engram"))
        config.ensure_dirs()
        config.set_value("top_k", "25")

        # Reload from file
        config2 = EngramConfig.load(data_dir=str(tmp_path / "engram"))
        assert config2.top_k == 25


class TestSaveLoad:
    """Tests for config persistence."""

    def test_save_and_reload(self, tmp_path):
        config = EngramConfig(
            data_dir=str(tmp_path / "engram"),
            decay_rate=0.02,
            top_k=15,
        )
        config.save()

        config2 = EngramConfig.load(data_dir=str(tmp_path / "engram"))
        assert config2.decay_rate == 0.02
        assert config2.top_k == 15
