import cv2
import numpy as np
import tensorrt as trt
from typing import List
import pycuda.driver as cuda
from pycuda import autoinit
from numpy.lib.stride_tricks import as_strided

from mot.structures import Detection
from mot.encode import ENCODER_REGISTRY, Encoder
from mot.detect import DETECTOR_REGISTRY, Detector

__all__ = ['TRTDGNetEncoder']

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# convert caffemodel to tensorrt model
class _CaffeModel(object):
    def __init__(self, model_info):
        self.model_info = model_info

    def _get_engine(self):
        # build engine based on caffe
        def _build_engine_caffe(model_info):
            def GiB(x):
                return x * 1 << 30

            with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
                builder.max_batch_size = model_info.max_batch_size
                builder.max_workspace_size = GiB(model_info.max_workspace_size)
                builder.fp16_mode = model_info.flag_fp16

                # Parse the model and build the engine.
                model_tensors = parser.parse(deploy=model_info.deploy_file, model=model_info.model_file,
                                             network=network,
                                             dtype=model_info.data_type)
                for ind_out in range(len(model_info.output_name)):
                    print(model_info.output_name[ind_out])
                    print(model_tensors.find(model_info.output_name[ind_out]))
                    network.mark_output(model_tensors.find(model_info.output_name[ind_out]))
                print("Building TensorRT engine. This may take a few minutes.")
                return builder.build_cuda_engine(network)

        try:
            with open(self.model_info.engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                print('-------------------load engine-------------------')
                return runtime.deserialize_cuda_engine(f.read())
        except:
            # Fallback to building an engine if the engine cannot be loaded for any reason.
            engine = _build_engine_caffe(self.model_info)
            with open(self.model_info.engine_file, "wb") as f:
                f.write(engine.serialize())
                print('-------------------save engine-------------------')
            return engine

    # allocate buffers
    def _allocate_buffers(self):
        engine = self._get_engine()
        h_output = []
        d_output = []
        h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),
                                        dtype=trt.nptype(self.model_info.data_type))
        # h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
        for ind_out in range(len(self.model_info.output_name)):
            h_output_temp = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(ind_out + 1)),
                                                  dtype=trt.nptype(self.model_info.data_type))
            h_output.append(h_output_temp)

        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        for ind_out in range(len(self.model_info.output_name)):
            d_output_temp = cuda.mem_alloc(h_output[ind_out].nbytes)
            d_output.append(d_output_temp)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        return engine.create_execution_context(), h_input, d_input, h_output, d_output, stream

    def get_outputs(self):
        context, h_input, d_input, h_output, d_output, stream = self._allocate_buffers()
        return context, h_input, d_input, h_output, d_output, stream


# process outputs of tensorrt
class _ResNetModel(object):
    def __init__(self, deploy_file, model_file, engine_file, input_shape=(3, 256, 128), output_name=None,
                 data_type=trt.float32, flag_fp16=True, max_workspace_size=1, max_batch_size=1, num_classes=1,
                 max_per_image=20):
        self.heat_shape = 128
        self.mate = None
        self.deploy_file = deploy_file
        self.model_file = model_file
        self.engine_file = engine_file
        self.data_type = data_type
        self.flag_fp16 = flag_fp16
        self.output_name = output_name
        self.max_workspace_size = max_workspace_size
        self.max_batch_size = max_batch_size
        self.input_shape = input_shape
        self.confidence = -1.0
        self.num_classes = num_classes
        self.max_per_image = max_per_image

    def do_inference(self, context, h_input, d_input, h_output, d_output, stream):
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        bindings = [int(d_input)]
        for ind_out in range(len(d_output)):
            bindings.append(int(d_output[ind_out]))
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.

        for ind_out in range(len(d_output)):
            cuda.memcpy_dtoh_async(h_output[ind_out], d_output[ind_out], stream)
        # Synchronize the stream
        stream.synchronize()


@ENCODER_REGISTRY.register()
class TRTDGNetEncoder(Encoder):
    def __init__(self, prototxt_path: str, model_path: str, engine_path: str, **kwargs):
        super().__init__()
        self.model = _ResNetModel(
            deploy_file=prototxt_path,
            model_file=model_path,
            engine_file=engine_path,
            input_shape=(3, 512, 512),
            output_name=['relu_blob49'],
            data_type=trt.float32,
            flag_fp16=True,
            max_workspace_size=1,
            max_batch_size=1,
        )
        caffeModel = _CaffeModel(self.model)
        self.size = (128, 256)
        self.context, self.h_input, self.d_input, self.h_output, self.d_output, self.stream = caffeModel.get_outputs()
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    # Extract feature
    def fliplr(self, img):
        return img[:, :, :, ::-1]

    def img_norm(self, img):
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.norm_mean[i]) / self.norm_std[i]
        return img

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = np.stack([self.img_norm(_resize(im, self.size)) for im in im_crops], axis=0)
        im_batch = im_batch.transpose((0, 3, 1, 2))
        return im_batch

    def feat_norm(self, f):
        # f = f.squeeze()
        fnorm = np.linalg.norm(f, ord=2, axis=1, keepdim=True)
        return f / fnorm

    def encode(self, detections: List[Detection], full_img: np.ndarray) -> List[object]:
        all_crops = []
        for detection in detections:
            box = detection.box
            crop = full_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if crop.shape[0] * crop.shape[1] > 0:
                all_crops.append(crop)
            else:
                all_crops.append(np.ones((10, 10, 3)).astype(np.float32) * 255)

        if len(detections) != 0:
            img = self._preprocess(all_crops)
            n, c, h, w = img.shape
            ff = np.zeros([n, 1024])
            for i in range(2):
                if (i == 1):
                    img = self.fliplr(img)

                self.model.do_inference(self.context, self.h_input, self.d_input, self.h_output, self.d_output,
                                        self.stream)
                f = self.h_output
                ff = ff + f

            ff[:, 0:512] = self.feat_norm(ff[:, 0:512])
            ff[:, 512:1024] = self.feat_norm(ff[:, 512:1024])
            return ff
        else:
            return []
