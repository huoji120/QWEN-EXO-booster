from pathlib import Path
import yaml

from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.policy_data import PolicyDataRepository

CORPUS_ROOT = Path(__file__).resolve().parents[3] / "scripts" / "qwen_exo" / "corpus"
AGENT_CONFIG = CORPUS_ROOT.parent / "mini_swe_agent_qwen_exo.yaml"
CODE_AGENT_CONFIG = CORPUS_ROOT.parent / "mini_swe_agent_qwen_exo_code.yaml"


def test_code_agent_ablation_disables_repetition_penalty_only():
    baseline = yaml.safe_load(AGENT_CONFIG.read_text(encoding="utf-8"))
    code_agent = yaml.safe_load(CODE_AGENT_CONFIG.read_text(encoding="utf-8"))

    baseline_sampling = baseline["model"]["model_kwargs"]
    code_sampling = code_agent["model"]["model_kwargs"]
    assert code_sampling == {
        **baseline_sampling,
        "extra_body": {
            **baseline_sampling["extra_body"],
            "repetition_penalty": 1.0,
        },
    }


def test_swe_agent_does_not_override_action_policy():
    config = yaml.safe_load(AGENT_CONFIG.read_text(encoding="utf-8"))

    assert not {
        "system_template",
        "instance_template",
        "step_limit",
        "whitelist_actions",
    }.intersection(config["agent"])
    assert not {
        "observation_template",
        "format_error_template",
        "tool_choice",
    }.intersection(config["model"])
    assert "environment" not in config
    assert config["agent"]["context_window_tokens"] == 102400
    assert "max_tokens" not in config["model"]["model_kwargs"]
    sampling = config["model"]["model_kwargs"]
    assert sampling["max_output_tokens"] == 8192
    assert sampling["temperature"] == 0.8
    assert sampling["top_p"] == 0.95
    assert sampling["extra_body"] == {
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
    }
    assert "frequency_penalty" not in sampling


def test_swe_corpus_is_answer_free_and_rankable(tmp_path):
    policy = PolicyDataRepository(CORPUS_ROOT / "policydata")
    knowledge = KnowledgeRepository(CORPUS_ROOT / "knowledge")
    policy_snapshot = policy.refresh()
    knowledge_snapshot = knowledge.refresh()

    assert len(policy_snapshot.documents) == 1
    expected_knowledge_paths = {
        "frontend-visual-quality-guardrails.md",
        "python-change-surfaces.md",
        "repository-evidence-and-change-verification.md",
        "threejs-architecture-and-assets.md",
        "threejs-lifecycle-and-verification.md",
        "threejs-performance-and-rendering.md",
        "wfp-appid-ale-auth-connect.md",
        "reflection-memory/execute-minimal-discriminating-probe.md",
        "reflection-memory/retrieval-review-injection-boundaries.md",
        "reflection-memory/separate-tool-failures-from-domain-evidence.md",
    }
    assert {
        document.relative_path for document in knowledge_snapshot.documents
    } == expected_knowledge_paths
    reference_documents = knowledge_snapshot.documents
    assert all(document.canonical for document in reference_documents)
    assert all(document.canonical for document in policy_snapshot.documents)
    reference_root = tmp_path / "reference-knowledge"
    for document in reference_documents:
        path = reference_root / document.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document.content, encoding="utf-8")
    reference_knowledge = KnowledgeRepository(reference_root)
    reference_knowledge.refresh()

    policy_query = (
        "Solve this repository issue on a branch, edit the source, run focused "
        "tests, commit the changes, and submit the final patch before timeout."
    )
    knowledge_query = (
        "Change a Python public API across wrappers, exports, generated paths, "
        "defaults, precedence rules, and regression tests."
    )
    ranked_policy = policy.rank(policy_query, limit=3)
    assert ranked_policy
    execution_policy = ranked_policy[0]
    assert execution_policy.relative_path == "coding-agent-execution-policy.md"
    assert "Recover from contradictory" in execution_policy.reference_content
    segmented_write_policy = policy.rank(
        "Create a substantive new source file, then implement several large sections.",
        limit=3,
    )[0]
    assert segmented_write_policy.relative_path == "coding-agent-execution-policy.md"
    opening_policy_tokens = " ".join(
        segmented_write_policy.normalized_reference_content.split()[:256]
    )
    assert "the first write must establish only a minimal valid skeleton" in (
        opening_policy_tokens
    )
    assert reference_knowledge.rank(knowledge_query, limit=3)
    assert not any(
        "trajectory" in document.relative_path.casefold()
        or "ctf" in document.relative_path.casefold()
        for document in reference_documents
    )
    attractor_cases = (
        (
            reference_knowledge,
            "WebGLRenderer GLTFLoader OrbitControls shaders draw calls GPU frame time",
            "threejs-architecture-and-assets.md",
        ),
        (
            reference_knowledge,
            "semantic HTML CSS Grid typography forms responsive focus accessibility",
            "frontend-visual-quality-guardrails.md",
        ),
        (
            policy,
            (
                "interactive rendered experience subject too small dark flat poorly "
                "framed generic camera composition lighting materials motion screenshots"
            ),
            "coding-agent-execution-policy.md",
        ),
        (
            policy,
            (
                "build a polished interactive 3D scene with animation camera controls "
                "lighting materials and responsive canvas"
            ),
            "coding-agent-execution-policy.md",
        ),
        (
            policy,
            "commands repeat without new evidence failed hypothesis malformed output timeout",
            "coding-agent-execution-policy.md",
        ),
        (
            policy,
            "explicit omitted defaults precedence ordered include exclude recursive",
            "coding-agent-execution-policy.md",
        ),
        (
            policy,
            "claim completion exact test exit status artifact report",
            "coding-agent-execution-policy.md",
        ),
        (
            policy,
            "authoritative implementation affected callers narrow patch request-scoped state",
            "coding-agent-execution-policy.md",
        ),
    )
    for repository, query, expected_path in attractor_cases:
        ranked = repository.rank(query, limit=3)
        assert ranked
        assert ranked[0].relative_path == expected_path
    visual_creation_query = (
        "build a polished interactive 3D scene with animation camera controls "
        "lighting materials and responsive canvas"
    )
    combined_visual = sorted(
        (
            *reference_knowledge.rank(visual_creation_query, limit=3),
            *policy.rank(visual_creation_query, limit=3),
        ),
        key=lambda candidate: -candidate.score,
    )
    assert any(
        candidate.relative_path == "coding-agent-execution-policy.md"
        for candidate in combined_visual
    )
    assert "By tool turn" not in execution_policy.reference_content
    assert "Hard tool-turn gates" not in execution_policy.reference_content
    visual_policy = policy_snapshot.documents[0].content.casefold()
    assert all(
        term not in visual_policy
        for term in (
            "car_factory",
            "pinball.html",
            "globe.html",
            "汽車工廠",
            "彈珠機",
            "3d 地球仪",
        )
    )
    cognition_root = CORPUS_ROOT / "cognition"
    assert not cognition_root.exists() or not tuple(cognition_root.glob("*.md"))
    execution_policy = next(
        document
        for document in policy_snapshot.documents
        if document.source_kind == "coding_agent_execution_policy"
    ).content
    personality_policy = execution_policy.casefold()
    assert "policydata is the authoritative personality document" in personality_policy
    assert "stable operational identity is gpt" in personality_policy
    assert "do not claim that conversation changes model weights" in personality_policy
    assert "repository analysis" in personality_policy
    assert "keep private chain-of-thought internal" in personality_policy

    corpus_text = "\n".join(
        document.content.casefold()
        for document in (*policy_snapshot.documents, *reference_documents)
    )
    assert "state machine" not in corpus_text
    assert "phase=" not in corpus_text
    assert "shell command" not in corpus_text
    assert "tool call" not in corpus_text
    assert "fixed procedure" not in corpus_text
    forbidden_benchmark_terms = {
        "cattrs",
        "fastapi",
        "partial_structure",
        "auto_head",
        "auto_options",
        "implicitmethodtrackingmiddleware",
    }
    assert not forbidden_benchmark_terms.intersection(corpus_text.split())
    assert all(term not in corpus_text for term in forbidden_benchmark_terms)
