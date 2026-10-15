# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_dspark_config(architecture: str) -> dict:
    return {
        "speculators_model_type": "dspark",
        "architectures": [architecture],
        "sample_from_anchor": True,
        "use_aux_hidden_state": True,
        "draft_vocab_size": 32,
        "target_hidden_size": 16,
        "mask_token_id": 31,
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "block_size": 3,
        "aux_hidden_state_layer_ids": [2, 5],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
    }


def test_dspark_updater_preserves_qwen3_omni_contract() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("Qwen3OmniDSparkModel")
    )

    assert config["architectures"] == ["Qwen3OmniDSparkModel"]
    assert config["sample_from_anchor"] is True
    assert config["dspark_bonus_anchor"] is False
    assert config["use_aux_hidden_state"] is True
    assert config["target_layer_ids"] == [1, 4]


def test_dspark_updater_keeps_legacy_qwen3_checkpoint_loadable() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("DSparkSpeculator")
    )

    assert config["architectures"] == ["Qwen3DSparkModel"]


def test_dspark_updater_maps_bonus_anchor_semantics() -> None:
    outer_config = _make_dspark_config("Qwen3DSparkModel")
    outer_config["sample_from_anchor"] = False

    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer_config)

    assert config["sample_from_anchor"] is False
    assert config["dspark_bonus_anchor"] is True


def test_mmuse_updater_preserves_identity_and_extension_options() -> None:
    outer_config = _make_dspark_config("MMuseSpeculator")
    outer_config.update(
        speculators_model_type="mmuse",
        enable_correction_head=True,
        correction_hidden_feedback=True,
        correction_output_mode="logits",
        dflash2_candidate_selector=True,
        dflash_gated_layer_fusion=True,
    )

    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer_config)

    assert config["model_type"] == "qwen3"
    assert config["architectures"] == ["MMuseSpeculator"]
    assert config["speculators_model_type"] == "mmuse"
    assert config["target_layer_ids"] == [1, 4]
    assert config["eagle_aux_hidden_state_layer_ids"] == [2, 5]
    assert config["mmuse_source_config"]["transformer_layer_config"] == {
        "model_type": "qwen3",
        "hidden_size": 16,
    }
    for key in (
        "enable_correction_head",
        "correction_hidden_feedback",
        "correction_output_mode",
        "dflash2_candidate_selector",
        "dflash_gated_layer_fusion",
    ):
        assert config[key] == outer_config[key]


def test_mmuse_speculative_config_keeps_its_own_method() -> None:
    config = _make_dspark_config("MMuseSpeculator")
    config["speculators_model_type"] = "mmuse"
    config["speculators_config"] = {
        "proposal_methods": [{"speculative_tokens": 3}],
        "verifier": {"name_or_path": "test/verifier"},
    }

    result = SpeculatorsConfig.extract_vllm_speculative_config(config)

    assert result == {"method": "mmuse", "num_speculative_tokens": 3}
