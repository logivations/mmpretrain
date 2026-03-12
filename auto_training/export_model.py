import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmpretrain.models import build_classifier

# Patch torch.load for PyTorch 2.6+ (weights_only=True breaks numpy in checkpoints)
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, 'weights_only': False})

try:
    import onnx
    import onnxruntime as rt
except ImportError as e:
    raise ImportError(f'Please install onnx and onnxruntime first. {e}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMPretrain models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='show onnx graph')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 3, 128, 128],
        help='input size')
    parser.add_argument(
        '--no-dynamic-batch',
        action='store_false',
        dest='dynamic_batch',
        help='disable dynamic batch size (enabled by default)')
    parser.set_defaults(dynamic_batch=True)
    return parser.parse_args()


def pytorch2onnx(model, input_shape, opset_version, output_file, verify,
                 show, dynamic_batch):
    """Export PyTorch model to ONNX format."""
    model.eval()

    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    print(f'Exporting model to ONNX with input shape: {input_shape}, '
          f'dynamic_batch={dynamic_batch}')
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        keep_initializers_as_inputs=False)

    print(f'Successfully exported ONNX model: {output_file}')

    if verify:
        print('Verifying ONNX model...')
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)

        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()

        sess = rt.InferenceSession(output_file)
        onnx_output = sess.run(None, {'input': dummy_input.numpy()})[0]

        if np.allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5):
            print('ONNX model verified: outputs match PyTorch!')
        else:
            print('Warning: ONNX outputs differ from PyTorch outputs')
            print(f'Max difference: {np.max(np.abs(pytorch_output - onnx_output))}')

    if show:
        onnx_model = onnx.load(output_file)
        print(onnx.helper.printable_graph(onnx_model.graph))


def export_model(config_path, checkpoint_path, output_dir=None,
                 input_shape=(1, 3, 128, 128), opset_version=11,
                 dynamic_batch=True, output_name='classifier.onnx'):
    """Export a trained MMPretrain classifier to ONNX format.

    Args:
        config_path: Path to the config file or Config object.
        checkpoint_path: Path to the checkpoint file.
        output_dir: Directory to save the exported model. Defaults to work_dir.
        input_shape: Input tensor shape (default: (1, 3, 128, 128)).
        opset_version: ONNX opset version (default: 11).
        dynamic_batch: Export with dynamic batch size axis (default: True).
        output_name: Output filename (default: 'classifier.onnx').

    Returns:
        str: Path to the exported ONNX model.
    """
    if isinstance(config_path, Config):
        cfg = config_path
    else:
        cfg = Config.fromfile(config_path)

    model = build_classifier(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Bind mode='tensor' + softmax to match mmdeploy export behaviour
    _fwd = model.forward
    model.forward = lambda x: F.softmax(_fwd(x, mode='tensor'), dim=1)

    if output_dir is None:
        output_dir = cfg.work_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_name)

    pytorch2onnx(
        model,
        input_shape=input_shape,
        opset_version=opset_version,
        output_file=output_file,
        verify=False,
        show=False,
        dynamic_batch=dynamic_batch)

    print(f'Model exported successfully to: {output_file}')
    return output_file


if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    _fwd = model.forward
    model.forward = lambda x: F.softmax(_fwd(x, mode='tensor'), dim=1)

    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    pytorch2onnx(
        model,
        input_shape=tuple(args.shape),
        opset_version=args.opset_version,
        output_file=args.output_file,
        verify=args.verify,
        show=args.show,
        dynamic_batch=args.dynamic_batch)
