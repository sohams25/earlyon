# Staged deployment

How to get a *real* early-exit compute saving outside of eager PyTorch.

## The three execution modes, honestly labelled

| Mode | Skips later layers? | Exportable? | Where |
|---|---|---|---|
| Eager wrapper (`model(x, mode="inference")`) | **yes** — hooks + a sentinel exception short-circuit the backbone | no (`torch.compile`/export refuse with a clear error) | `earlyon.core.wrappers` |
| Static multi-output ONNX (`earlyon export`, `export_to_onnx`) | **no** — the graph always computes every exit; the caller applies routing to the outputs | yes, single portable graph | `earlyon.onnx` |
| Staged deployment | **yes** — stages are separate modules; routing runs between them, later stages are never invoked | yes, one artifact per stage | `earlyon.staged` |

## The protocol

An early-exit model with `k` exits becomes `k + 1` ordered **stages**.

- Stage `i` (`0 <= i < k`) consumes the previous stage's **continuation
  features** (stage 0 consumes the input) and produces
  `(continuation_features, exit_logits_i)` where `exit_logits_i` comes from
  exit head `i`.
- Stage `k` (final) produces the final classifier's logits.
- Between stages the runtime applies exactly earlyon's routing rule, in this
  order: if `enabled_exits[i]` is false, skip the head and continue; else
  compute `softmax(logits / temperatures[exit_name])` and fire on
  `confidence >= confidence_thresholds[i]` (or entropy `<= threshold` for the
  entropy policy). The first firing exit's logits are the prediction; later
  stages are not executed.

The data classes live in `earlyon.staged`: `StageSpec` (index, exit name,
covered modules), `Stage` (trunk + head, `forward -> (features, logits)`),
and `StagedModel` (the reference runtime, `infer(x) -> InferenceResult`).

## Reference implementation and its scope

`earlyon.staged.staged_model(wrapper)` splits a wrapper whose backbone is
literally an `nn.Sequential` (exit layers at top level, `final_classifier`
identity — what `custom_ee` produces for a Sequential backbone). It refuses
anything else: arbitrary graph partitioning is out of scope, and a wrong
split is worse than no split. As a belt, the builder runs a probe input
through both the staged and eager paths and raises on any disagreement;
`tests/test_staged.py` pins equivalence across policies, per-head
temperatures and enablement.

```python
from earlyon.models import custom_ee
from earlyon.staged import staged_model

wrapper = custom_ee(sequential_backbone, exit_layers=["2", "5"], num_classes=10)
staged = staged_model(wrapper)          # raises if the split can't be proven
result = staged.infer(x)                # same InferenceResult as the wrapper
```

For the torchvision factories (ResNet, MobileNet, ViT), the backbone forward
is not a plain Sequential; implementing their stage splits is future work —
the protocol above is what such an implementation must satisfy, and the
equivalence test is the acceptance criterion.

## Exporting stages

Each `Stage` is a plain `nn.Module` with tensor-in/tensors-out, so it traces
cleanly with `torch.onnx.export` (outputs: `features`, `logits`). Your
deployment runtime (ONNX Runtime sessions, or TensorRT engines built from the
per-stage ONNX files) then implements the routing loop above on the host —
one engine invocation per stage until an exit fires.

## Jetson / TensorRT procedure (not run here — no fabricated numbers)

On the target device:

1. Export per-stage ONNX files from a calibrated checkpoint.
2. `trtexec --onnx=stage_k.onnx --saveEngine=stage_k.plan --fp16` per stage.
3. Runtime loop: run stage 0's engine, apply the threshold on the host
   (logits are tiny — the copy is negligible), continue only if no fire.
4. Benchmark with the same discipline as `earlyon.benchmarking`: fixed
   sample set, ≥50-iteration warmup, batch 1, report p50/p95/p99 and the
   exit distribution; compare against a single-engine full-backbone build
   *and* a smaller static model at matched accuracy.

Nothing in this repository claims TensorRT numbers; the section above is the
procedure, to be filled in from a real device run.
