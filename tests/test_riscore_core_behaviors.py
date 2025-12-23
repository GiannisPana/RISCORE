from __future__ import annotations


def make_example(question: str, answer_idx: int = 0, reconstructed: bool = True):
    from riscore.data import RiddleExample

    return RiddleExample(
        question=question,
        choices=["alpha", "beta", "gamma", "delta"],
        answer=["alpha", "beta", "gamma", "delta"][answer_idx],
        answer_idx=answer_idx,
        reconstructed_question=f"Reconstructed {question}" if reconstructed else None,
        reconstructed_choices=["one", "two", "three", "four"] if reconstructed else None,
        reconstructed_answer="one" if reconstructed else None,
        reconstructed_answer_idx=0 if reconstructed else None,
        cot=f"Reasoning for {question}",
    )


def test_unified_model_interface_create_builds_config(monkeypatch):
    from riscore.core import UnifiedModelInterface

    def fake_initialize(self):
        self.model = object()

    monkeypatch.setattr(UnifiedModelInterface, "_initialize_model", fake_initialize)

    model = UnifiedModelInterface.create(
        model_name="gpt-4o-mini",
        provider="openai",
        api_key="test-key",
        temperature=0.2,
        max_tokens=128,
    )

    assert model.config.provider == "openai"
    assert model.config.model_name == "gpt-4o-mini"
    assert model.config.api_key == "test-key"
    assert model.generation_config.temperature == 0.2
    assert model.generation_config.max_new_tokens == 128


def test_riscore_prompt_uses_selected_exemplar_count(monkeypatch):
    from riscore.prompting import RISCOREPrompt

    prompt = RISCOREPrompt(similarity_based_selection=False)
    prompt.similarity_based_selection = True
    pool = [make_example(f"exemplar {idx}") for idx in range(4)]
    target = make_example("target", reconstructed=False)

    monkeypatch.setattr(
        prompt,
        "find_similar_exemplars",
        lambda target_example, candidate_examples, k=None, **kwargs: candidate_examples[1 : 1 + k],
    )

    formatted = prompt.format_prompt(target, exemplars=pool, num_exemplars=2)

    assert "exemplar 1" in formatted
    assert "exemplar 2" in formatted
    assert "exemplar 0" not in formatted
    assert "exemplar 3" not in formatted


def test_riddle_example_round_trips_reconstructed_answer_idx():
    from riscore.data import RiddleExample

    example = make_example("round trip")
    serialized = example.to_dict()
    restored = RiddleExample.from_dict(serialized)

    assert serialized["reconstructed_answer_idx"] == 0
    assert restored.reconstructed_answer_idx == 0


def test_context_reconstruction_parses_valid_response():
    from riscore.prompting import ContextReconstructionGenerator

    class FakeModel:
        def generate_chat(self, **kwargs):
            return """Question: I speak without a mouth. What am I?
A. echo
B. clock
C. river
D. paper
Answer: A"""

    generator = ContextReconstructionGenerator(FakeModel())
    result = generator.generate_reconstruction(make_example("original"), validate=True)

    assert result["reconstructed_question"].startswith("I speak")
    assert result["reconstructed_answer"] == "echo"
    assert result["reconstructed_answer_idx"] == 0
