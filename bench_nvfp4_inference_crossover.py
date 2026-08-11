from __future__ import annotations

import json
import statistics
import time

import modal

from chess_engine_4.modal_train import app, image


@app.function(image=image, gpu="B200", cpu=8, timeout=60 * 60)
def benchmark() -> dict[str, object]:
    import torch

    from chess_engine_4.data.leela import INPUT_PLANE_COUNT, POLICY_SIZE
    from chess_engine_4.model import DenseChessNetConfig, build_model
    from chess_engine_4.model.transformer_engine import autocast_context

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.set_float32_matmul_precision("high")

    def summarize(samples: list[float]) -> dict[str, float]:
        ordered = sorted(samples)
        return {
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "p10_ms": ordered[max(0, round(0.1 * (len(ordered) - 1)))],
            "p90_ms": ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))],
        }

    def measure_paired(functions, *, warmup: int, iterations: int):
        names = tuple(functions)
        for index in range(warmup):
            order = names if index % 2 == 0 else tuple(reversed(names))
            for name in order:
                functions[name]()
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for index in range(iterations):
            order = names if index % 2 == 0 else tuple(reversed(names))
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                functions[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        return {name: summarize(values) for name, values in samples.items()}

    def make_model(precision: str, d_model: int):
        config = DenseChessNetConfig(
            d_model=d_model,
            depth=8,
            expansion_ratio=4.0,
            activation="swiglu",
            precision=precision,
            kernel_backend="te",
        )
        return build_model(config).cuda()

    def cached_forward(model, planes, *, first_microbatch: bool):
        x = planes.flatten(start_dim=1)
        x = model.input(x, is_first_microbatch=first_microbatch)
        for block in model.blocks:
            x = x + block.layer(x, is_first_microbatch=first_microbatch)
        x = model.norm(x)
        return (
            model.policy_head(x, is_first_microbatch=first_microbatch)[:, :POLICY_SIZE],
            model.wdl_head(x, is_first_microbatch=first_microbatch)[:, :3],
            model.moves_left_head(x, is_first_microbatch=first_microbatch)[:, 0],
        )

    result: dict[str, object] = {
        "device": torch.cuda.get_device_name(),
        "transformer_engine": __import__("transformer_engine").__version__,
        "torch": torch.__version__,
        "depth": 8,
        "expansion_ratio": 4.0,
        "activation": "swiglu",
        "batch_size": 4096,
        "widths": {},
    }

    precisions = ("mxfp8", "nvfp4")
    batch_size = 4096
    for d_model in (3072, 4096):
        inference_models = {}
        width_result = {}
        for precision in precisions:
            torch.manual_seed(1)
            inference_models[precision] = make_model(precision, d_model).eval()
        planes = torch.randn(
            batch_size,
            INPUT_PLANE_COUNT,
            8,
            8,
            device="cuda",
            dtype=torch.bfloat16,
        )
        functions = {}
        for precision, model in inference_models.items():
            with torch.inference_mode(), autocast_context(precision):
                cached_forward(model, planes, first_microbatch=True)

            def run_cached(precision=precision, model=model, planes=planes):
                with torch.inference_mode(), autocast_context(precision):
                    cached_forward(model, planes, first_microbatch=False)

            functions[precision] = run_cached
        timings = measure_paired(functions, warmup=10, iterations=30)
        for precision, timing in timings.items():
            timing["positions_per_second"] = batch_size / (timing["median_ms"] / 1000)
            width_result[precision] = timing
        width_result["nvfp4_speedup"] = (
            timings["mxfp8"]["median_ms"] / timings["nvfp4"]["median_ms"]
        )
        result["widths"][str(d_model)] = width_result
        del planes, functions, inference_models
        torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    started = time.time()
    with modal.enable_output(), app.run():
        output = benchmark.remote()
    output["wall_seconds"] = time.time() - started
    print("BENCHMARK_JSON=" + json.dumps(output, sort_keys=True))
