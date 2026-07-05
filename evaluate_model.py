import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image, ImageOps


def workaround_old_model(model_path: str) -> None:
    """Remove legacy unsupported 'groups' config from some Teachable Machine exports."""
    with h5py.File(model_path, mode="r+") as model_file:
        model_config = model_file.attrs.get("model_config")
        if model_config is None:
            return

        if isinstance(model_config, bytes):
            model_config = model_config.decode("utf-8")

        if '"groups": 1,' in model_config:
            model_config = model_config.replace('"groups": 1,', "")
            model_file.attrs.modify("model_config", model_config)
            model_file.flush()


def read_class_names(labels_path: str) -> list[str]:
    class_names: list[str] = []
    with open(labels_path, "r", encoding="utf-8") as labels_file:
        for raw_line in labels_file:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                class_names.append(parts[1].strip())
            else:
                class_names.append(line)

    if not class_names:
        raise ValueError(f"No classes found in labels file: {labels_path}")

    return class_names


def prepare_image(image_path: Path, width: int, height: int, use_teachable_machine_norm: bool) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)

    if use_teachable_machine_norm:
        image_array = (image_array / 127.5) - 1

    data = np.expand_dims(image_array, axis=0)
    return data


def normalize_root_dir_value(raw_root: str) -> str:
    return raw_root.strip().strip('"').strip("'")


def evaluate_model(config_path: str, 
                   model_path: str, 
                   labels_path: str, 
                   output_path: str ) -> str:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    #mot needed normally, done once for all 
    # workaround_old_model(model_path)

    model = load_model(model_path, compile=False)
    class_names = read_class_names(labels_path)

    #TODO find a solution without warning
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 4 or input_shape[1] is None or input_shape[2] is None:
        raise ValueError(f"Unsupported model input shape: {input_shape}")

    img_height = int(input_shape[1])
    img_width = int(input_shape[2])
    use_tm_norm = img_height == 224 and img_width == 224

    tree = ET.parse(config_file)
    config_root = tree.getroot()

    root_dir_node = config_root.find("root_dir")
    root_dir_text = root_dir_node.text if root_dir_node is not None and root_dir_node.text else ""
    normalized_root_dir = normalize_root_dir_value(root_dir_text)

    result_root = ET.Element("test_result")
    result_root_dir = ET.SubElement(result_root, "root_dir")
    result_root_dir.text = root_dir_text

    success_node = ET.SubElement(result_root, "success")
    fail_node = ET.SubElement(result_root, "fail")

    total = 0
    passed = 0

    for test_item in config_root.findall("test_item"):
        file_name = test_item.get("file", "")
        expected = test_item.get("expected", "")

        total += 1

        image_path = Path(normalized_root_dir) / file_name if normalized_root_dir else Path(file_name)
        if not image_path.exists():
            image_path = config_file.parent / file_name

        if not image_path.exists():
            ET.SubElement(
                fail_node,
                "test_item",
                {
                    "file": file_name,
                    "expected": expected,
                    "got": "FILE_NOT_FOUND",
                    "confidence_score": "0.00",
                },
            )
            continue

        data = prepare_image(image_path, img_width, img_height, use_tm_norm)
        prediction = model.predict(data, verbose=0)

        if prediction.ndim == 2:
            scores = prediction[0]
        else:
            scores = np.ravel(prediction)

        if np.any(scores < 0) or np.sum(scores) > 1.01:
            probs = tf.nn.softmax(scores).numpy()
        else:
            probs = scores

        index = int(np.argmax(probs))
        got = class_names[index] if index < len(class_names) else f"class_{index}"
        confidence = float(probs[index]) * 100.0

        attrs = {
            "file": file_name,
            "expected": expected,
            "got": got,
            "confidence_score": f"{confidence:.2f}",
        }

        if got == expected:
            ET.SubElement(success_node, "test_item", attrs)
            passed += 1
        else:
            ET.SubElement(fail_node, "test_item", attrs)

    accuracy = (passed / total * 100.0) if total else 0.0
    summary = ET.SubElement(result_root, "summary")
    summary.set("total", str(total))
    summary.set("passed", str(passed))
    summary.set("failed", str(total - passed))
    summary.set("accuracy", f"{accuracy:.2f}")

    if output_path is None:
        if config_file.suffix.lower() == ".xml":
            output_file = str(config_file.with_name(config_file.stem + "-result.xml"))
        else:
            output_file = str(config_file.with_name(config_file.name + "-result.xml"))
    else:
        output_file = output_path

    ET.indent(result_root, space="    ")
    result_tree = ET.ElementTree(result_root)
    result_tree.write(output_file, encoding="utf-8", xml_declaration=True)

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Keras model with an XML config and produce a test_result XML file."
    )
    parser.add_argument(
        "-config",
        default="config-accuracy-test.xml",
        help="Path to input configuration XML file.",
    )
    parser.add_argument(
        "-model",
        default="keras_model.h5",
        help="Path to Keras model file.",
    )
    parser.add_argument(
        "-labels",
        default="labels.txt",
        help="Path to labels file.",
    )
    parser.add_argument(
        "-output",
        default=None,
        help="Path to output XML result. Default: <config-name>-result.xml",
    )

    args = parser.parse_args()

    output_file = evaluate_model(
        config_path=args.config,
        model_path=args.model,
        labels_path=args.labels,
        output_path=args.output,
    )

    print(f"Test result written to: {output_file}")


if __name__ == "__main__":
    main()
