import io
import os
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
from PIL import Image, ImageOps

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            raise ImportError(
                "No TensorFlow Lite runtime found. Please install `ai-edge-litert` or `tflite-runtime`."
            )

BASE_DIR = Path(__file__).resolve().parent

_INTERPRETER_CACHE: Optional[Any] = None
_INPUT_DETAILS: Optional[List[Dict[str, Any]]] = None
_OUTPUT_DETAILS: Optional[List[Dict[str, Any]]] = None
_LABELS_CACHE: Optional[List[str]] = None


def find_model_path() -> Path:
    model_file = BASE_DIR / "model_quant.tflite"
    if model_file.exists():
        return model_file
    raise FileNotFoundError(
        f"TensorFlow Lite model file (model_quant.tflite) not found in {BASE_DIR}."
    )


def find_labels_path() -> Path:
    labels_file = BASE_DIR / "labels.txt"
    if labels_file.exists():
        return labels_file
    raise FileNotFoundError(
        f"labels.txt file not found in {BASE_DIR}."
    )


def load_labels(labels_path: Optional[Path] = None) -> List[str]:
    global _LABELS_CACHE
    if _LABELS_CACHE is None:
        path = labels_path or find_labels_path()
        if not path.exists():
            raise FileNotFoundError(f"labels.txt not found at {path}")

        labels_dict: Dict[int, str] = {}
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels_dict[int(parts[0])] = parts[1].strip()
                else:
                    labels_dict[idx] = line

        if not labels_dict:
            raise ValueError(f"No labels found in {path}")

        max_idx = max(labels_dict.keys())
        _LABELS_CACHE = [labels_dict.get(i, f"Class_{i}") for i in range(max_idx + 1)]
    return _LABELS_CACHE


def load_interpreter() -> Any:
    global _INTERPRETER_CACHE, _INPUT_DETAILS, _OUTPUT_DETAILS
    if _INTERPRETER_CACHE is None:
        model_path = find_model_path()
        try:
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            _INPUT_DETAILS = interpreter.get_input_details()
            _OUTPUT_DETAILS = interpreter.get_output_details()
            _INTERPRETER_CACHE = interpreter
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow Lite model from {model_path}: {e}")
    return _INTERPRETER_CACHE


def get_input_details() -> List[Dict[str, Any]]:
    load_interpreter()
    return _INPUT_DETAILS or []


def get_output_details() -> List[Dict[str, Any]]:
    load_interpreter()
    return _OUTPUT_DETAILS or []


def preprocess_image(
    image_input: Union[str, Path, bytes, Image.Image, io.BytesIO],
    target_shape: List[int],
    input_dtype: np.dtype,
    quantization: tuple
) -> np.ndarray:
    try:
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        img = img.convert("RGB")
    except Exception as img_err:
        raise ValueError(f"Invalid or unreadable image: {img_err}")

    if len(target_shape) == 4:
        height, width = target_shape[1], target_shape[2]
    else:
        height, width = 224, 224

    img = ImageOps.fit(img, (width, height), Image.Resampling.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32)

    scale, zero_point = quantization if quantization else (0.0, 0)

    if input_dtype == np.float32:
        normalized_array = (img_array / 127.5) - 1.0
        final_array = normalized_array.astype(np.float32)
    elif input_dtype == np.uint8:
        if scale > 0:
            final_array = np.round((img_array / 255.0) / scale + zero_point).clip(0, 255).astype(np.uint8)
        else:
            final_array = img_array.astype(np.uint8)
    elif input_dtype == np.int8:
        if scale > 0:
            final_array = np.round(((img_array / 127.5) - 1.0) / scale + zero_point).clip(-128, 127).astype(np.int8)
        else:
            final_array = (img_array - 128).clip(-128, 127).astype(np.int8)
    else:
        final_array = img_array.astype(input_dtype)

    return np.expand_dims(final_array, axis=0)


def predict_image(image_input: Union[str, Path, bytes, Image.Image, io.BytesIO]) -> Dict[str, Any]:
    interpreter = load_interpreter()
    input_details = _INPUT_DETAILS[0]
    output_details = _OUTPUT_DETAILS[0]
    labels = load_labels()

    target_shape = input_details["shape"].tolist()
    input_dtype = input_details["dtype"]
    quantization = input_details.get("quantization", (0.0, 0))

    input_data = preprocess_image(
        image_input=image_input,
        target_shape=target_shape,
        input_dtype=input_dtype,
        quantization=quantization
    )

    try:
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])[0]
    except Exception as infer_err:
        raise RuntimeError(f"TensorFlow Lite inference failed: {infer_err}")

    out_scale, out_zero_point = output_details.get("quantization", (0.0, 0))
    if output_details["dtype"] in (np.uint8, np.int8) and out_scale > 0:
        scores = out_scale * (raw_output.astype(np.float32) - out_zero_point)
    else:
        scores = raw_output.astype(np.float32)

    sum_scores = float(np.sum(scores))
    if np.isclose(sum_scores, 1.0, atol=1e-2) and np.all(scores >= 0):
        probs = scores
    else:
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)

    predicted_idx = int(np.argmax(probs))
    category = labels[predicted_idx] if predicted_idx < len(labels) else f"Category_{predicted_idx}"
    confidence = float(probs[predicted_idx])

    probabilities_dict = {}
    for i, prob in enumerate(probs):
        lbl = labels[i] if i < len(labels) else f"Class_{i}"
        probabilities_dict[lbl] = round(float(prob), 4)

    return {
        "class_index": predicted_idx,
        "category": category,
        "confidence": round(confidence, 4),
        "probabilities": probabilities_dict
    }


if __name__ == "__main__":
    print("=" * 60)
    print("FLIPKART TENSORFLOW LITE IMAGE CLASSIFIER TEST")
    print("=" * 60)
    try:
        model_p = find_model_path()
        labels_p = find_labels_path()
        print(f"Model file: {model_p.name}")
        print(f"Labels file: {labels_p.name}")
        loaded_labels = load_labels()
        print(f"Detected Labels: {loaded_labels}")

        interp = load_interpreter()
        inp_shape = _INPUT_DETAILS[0]['shape'].tolist()
        inp_dtype = _INPUT_DETAILS[0]['dtype'].__name__
        out_shape = _OUTPUT_DETAILS[0]['shape'].tolist()
        out_dtype = _OUTPUT_DETAILS[0]['dtype'].__name__
        print(f"Input shape: {inp_shape} | Input dtype: {inp_dtype}")
        print(f"Output shape: {out_shape} | Output dtype: {out_dtype}")

        test_img = Image.new("RGB", (224, 224), color=(100, 180, 240))
        result = predict_image(test_img)
        print(f"\nSample Inference Result:")
        print(f"Predicted Category: {result['category']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print(f"Probabilities: {result['probabilities']}")
        print("=" * 60)
    except Exception as e:
        print(f"Error during TFLite test: {e}")
