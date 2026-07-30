from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper


WEIGHT_ROOT = "model_weights"


def _read(handle: h5py.File, layer: str, name: str) -> np.ndarray:
    return np.asarray(
        handle[f"{WEIGHT_ROOT}/{layer}/sequential/{layer}/{name}"],
        dtype=np.float32,
    )


def _initializer(name: str, values: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def build_model(source: Path, output: Path) -> None:
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        _initializer("input_scale", np.asarray(1.0 / 255.0, dtype=np.float32))
    ]
    nodes.append(helper.make_node("Mul", ["image", "input_scale"], ["scaled"]))
    nodes.append(
        helper.make_node("Transpose", ["scaled"], ["nchw"], perm=[0, 3, 1, 2])
    )
    current = "nchw"

    with h5py.File(source, "r") as handle:
        for index, (conv_name, batch_norm_name) in enumerate(
            (
                ("conv2d", "batch_normalization"),
                ("conv2d_1", "batch_normalization_1"),
                ("conv2d_2", "batch_normalization_2"),
            )
        ):
            kernel = _read(handle, conv_name, "kernel").transpose(3, 2, 0, 1)
            values = {
                "kernel": kernel,
                "bias": _read(handle, conv_name, "bias"),
                "gamma": _read(handle, batch_norm_name, "gamma"),
                "beta": _read(handle, batch_norm_name, "beta"),
                "mean": _read(handle, batch_norm_name, "moving_mean"),
                "variance": _read(handle, batch_norm_name, "moving_variance"),
            }
            for suffix, value in values.items():
                initializers.append(_initializer(f"block{index}_{suffix}", value))

            conv = f"block{index}_conv"
            activated = f"block{index}_relu"
            normalized = f"block{index}_batch_norm"
            pooled = f"block{index}_pool"
            nodes.extend(
                (
                    helper.make_node(
                        "Conv",
                        [current, f"block{index}_kernel", f"block{index}_bias"],
                        [conv],
                        pads=[1, 1, 1, 1],
                        strides=[1, 1],
                    ),
                    helper.make_node("Relu", [conv], [activated]),
                    helper.make_node(
                        "BatchNormalization",
                        [
                            activated,
                            f"block{index}_gamma",
                            f"block{index}_beta",
                            f"block{index}_mean",
                            f"block{index}_variance",
                        ],
                        [normalized],
                        epsilon=0.001,
                    ),
                    helper.make_node(
                        "MaxPool",
                        [normalized],
                        [pooled],
                        kernel_shape=[2, 2],
                        strides=[2, 2],
                    ),
                )
            )
            current = pooled

        dense_values = {
            "dense_kernel": _read(handle, "dense", "kernel"),
            "dense_bias": _read(handle, "dense", "bias"),
            "output_kernel": _read(handle, "dense_1", "kernel"),
            "output_bias": _read(handle, "dense_1", "bias"),
        }

    for name, value in dense_values.items():
        initializers.append(_initializer(name, value))

    nodes.extend(
        (
            helper.make_node("GlobalAveragePool", [current], ["global_average"]),
            helper.make_node("Flatten", ["global_average"], ["features"], axis=1),
            helper.make_node("MatMul", ["features", "dense_kernel"], ["dense_mm"]),
            helper.make_node("Add", ["dense_mm", "dense_bias"], ["dense_pre"]),
            helper.make_node("Relu", ["dense_pre"], ["dense_output"]),
            helper.make_node(
                "MatMul", ["dense_output", "output_kernel"], ["class_mm"]
            ),
            helper.make_node("Add", ["class_mm", "output_bias"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
        )
    )
    graph = helper.make_graph(
        nodes,
        "tomatoguard_legacy_onnx",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [None, 128, 128, 3])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 10])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="tomatoguard-export-legacy-onnx",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)
    print(f"Exported {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the tracked legacy Keras H5 graph to lightweight ONNX."
    )
    parser.add_argument("--source", type=Path, default=Path("model/tomatoes.h5"))
    parser.add_argument("--output", type=Path, default=Path("model/tomatoes.onnx"))
    args = parser.parse_args()
    build_model(args.source, args.output)


if __name__ == "__main__":
    main()
