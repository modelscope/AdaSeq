from modelscope.exporters.torch_model_exporter import TorchModelExporter


class Exporter(TorchModelExporter):
    """The base class of exporter inheriting from TorchModelExporter.

    This class provides the default implementations for exporting onnx and torch script.
    Each specific model may implement its own exporter by overriding the export_onnx/export_torch_script,
    and to provide implementations for generate_dummy_inputs/inputs/outputs methods.
    """

    def __init__(self, model=None, preprocessor=None):
        super().__init__(model=model)
        self.preprocessor = preprocessor
