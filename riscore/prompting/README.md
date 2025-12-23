# Prompting

Prompting contains the baseline strategies and the RISCORE-specific prompt
builder.

## Baselines

- `ZeroShotPrompt`
- `ZeroShotCoTPrompt`
- `FewShotPrompt`
- `FewShotCoTPrompt`
- `SelfConsistencyPrompt`
- `ActivePrompt`

## RISCORE

`RISCOREPrompt` formats exemplars as original/reconstructed riddle pairs. This
pushes the model toward the shared reasoning logic between the two contexts.

```python
from riscore.prompting import RISCOREPrompt

strategy = RISCOREPrompt(
    use_cot=True,
    num_exemplars=2,
    similarity_based_selection=True,
)
prompt = strategy.format_prompt(target_example, exemplars=train_examples)
```

`ContextReconstructionGenerator` uses a model to generate reconstructed versions
of training riddles:

```python
from riscore.prompting import ContextReconstructionGenerator

generator = ContextReconstructionGenerator(model, temperature=0.7)
augmented = generator.augment_dataset(train_examples)
```

YAML templates live in `riscore/prompting/templates/`.
