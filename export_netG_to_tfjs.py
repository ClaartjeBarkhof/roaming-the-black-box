import onnxruntime as rt
import onnx
from onnx2keras import onnx_to_keras
import tensorflowjs as tfjs

import torch
import matplotlib.pyplot as plt


def export_netG_to_tfjs(netG, tfjs_dir, onnx_save_path="netG.onnx", plot_checks=True):
    # -------------------------------------
    # PyTorch -> ONNX
    input_names = ["latent_batch"]
    output_names = ["image_batch"]

    dummy_input = torch.randn((1, 100, 1, 1))

    for param in netG.parameters():
        param.requires_grad = False

    torch.onnx.export(
        netG.to("cpu"),
        dummy_input,
        onnx_save_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names)

    if plot_checks:
        # Check if it has worked
        sess = rt.InferenceSession(onnx_save_path)
        input_name = sess.get_inputs()[0].name

        # Note: The input must be of the same shape as the shape of x during # the model
        # export part. i.e. second argument in this function call: torch.onnx.export()
        onnx_preds = sess.run(None, {input_name: dummy_input.numpy()})[0]

        print("predictions shape", onnx_preds.shape)
        print('input_name', input_name)

        plt.imshow(onnx_preds[0].transpose(1, 2, 0), vmin=0.0, vmax=1.0)
        plt.show()

    # -------------------------------------
    # ONNX -> Keras

    # Load ONNX model
    onnx_model = onnx.load(onnx_save_path)

    # Call the converter (input will be equal to the input_names parameter that you defined during exporting)
    k_model = onnx_to_keras(onnx_model, ['latent_batch'])

    if plot_checks:
        # print model summary to check
        print(k_model.summary())

        # check if it gives the same output plot
        keras_output = k_model.predict(dummy_input.numpy())
        print("keras_output.min(), keras_output.max()", keras_output.min(), keras_output.max())
        plt.imshow(keras_output[0].transpose(1, 2, 0))
        plt.show()

    # -------------------------------------
    # ONNX -> Keras
    k_model.compile()
    tfjs.converters.save_keras_model(k_model, tfjs_dir)
    print("Saved resulting files to:")
    print("onnx model:", onnx_save_path)
    print("tfjs:", tfjs_dir)