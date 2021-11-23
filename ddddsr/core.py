import io
import os
import math
import cv2
import onnxruntime
from PIL import Image
import numpy as np
from typing import Optional, Union, List, Tuple


class SR(object):
    def __init__(self, 
                 model: str = 'waifu2x_art',
                 scale: Optional[float] = None,
                 size: Optional[Union[int, List[int], Tuple[int]]] = None,
                 denoise_level: int =  2, # [-1, 3], -1 means no denoise
                 use_gpu: bool = False, 
                 device_id: int = 0,
                 ocr_text: bool = False,
                 ocr_font_size: int = 28,
                 ocr_font_color: Union[Tuple[int], List[int]] = (0, 0, 0),
                 ocr_background_color: Union[Tuple[int], List[int]] = (255, 255, 255),
                 ocr_font_ttf: Optional[str] = None,
                ):

        self.scale = scale
        self.size = size
        self.model = model
        self.denoise_level = denoise_level
        self.ocr_text = ocr_text
        self.ocr_font_size = ocr_font_size
        self.ocr_font_color = ocr_font_color
        self.ocr_background_color = ocr_background_color
        self.ocr_font_ttf = ocr_font_ttf

        if model == 'waifu2x_art':
            if denoise_level < 0:
                model_path = f'cunet/scale2.0x_model.onnx'
            else:
                model_path = f'cunet/noise{denoise_level}_scale2.0x_model.onnx'
        elif model == 'waifu2x_photo':
            if denoise_level < 0:
                model_path = f'upconv_7/scale2.0x_model.onnx'
            else:
                model_path = f'upconv_7/noise{denoise_level}_scale2.0x_model.onnx'
        else:
            raise NotImplementedError

        self.__model_path = os.path.join(os.path.dirname(__file__), 'models', model_path)

        if use_gpu:
            self.__providers = [
                ('CUDAExecutionProvider', {
                    'device_id': device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cuda_mem_limit': 8 * 1024 * 1024 * 1024, # 8GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
            ]
        else:
            self.__providers = [
                'CPUExecutionProvider',
            ]

        self.__ort_session = onnxruntime.InferenceSession(self.__model_path, providers=self.__providers)
        self.__ort_input_name = self.__ort_session.get_inputs()[0].name

        if self.ocr_text:
            from . import ppocronnx
            self.OCR = ppocronnx.TextSystem(use_gpu=use_gpu, device_id=device_id)

    def run(self, 
            image: np.ndarray,
            window_size: int = 128,
           ):
        
        # image: [H, W, 3] or [H, W, 4]

        # determine out size
        in_size = np.array(image.shape[:2])
        if self.scale is not None:
            out_size = (in_size * self.scale).astype(int)
        elif self.size is not None:
            if isinstance(self.size, int):
                out_size = np.array([self.size, self.size])
            else:
                out_size = np.array(self.size)
        else:
            # default is 2x
            out_size = in_size * 2

        # determine run times
        iter_2x = math.ceil(np.log2(out_size / in_size).max())

        # process image
        rgb = image[:, :, :3]
        alpha = image[:, :, 3] if image.shape[2] == 4 else None
        
        rgb = rgb.transpose(2,0,1).astype(np.float32) / 255

        # run
        if self.model == 'waifu2x_art':
            padding = 18
        elif self.model == 'waifu2x_photo':
            padding = 7
        else:
            padding = 0

        x = np.expand_dims(rgb, axis=0) # [1, 3, H, W]
        while iter_2x:
            if window_size == -1:
                x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
                x = self.__ort_session.run(None, {self.__ort_input_name: x})[0]
            else:
                h, w = x.shape[2], x.shape[3]
                pad_x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
                y =  np.zeros((1, 3, h * 2, w * 2))
                cnt = np.zeros((h * 2, w * 2))
                for h0 in range(0, h, window_size):
                    for w0 in range(0, w, window_size):
                        h1 = min(h, h0 + window_size)
                        w1 = min(w, w0 + window_size)
                        h0 = min(h0, h1 - window_size)
                        w0 = min(w0, w1 - window_size)
                        patch_x = pad_x[:, :, h0:h1+padding*2, w0:w1+padding*2]
                        patch_y = self.__ort_session.run(None, {self.__ort_input_name: patch_x})[0]
                        y[:, :, h0*2:h1*2, w0*2:w1*2] += patch_y
                        cnt[h0*2:h1*2, w0*2:w1*2] += 1
                x = y / cnt

            iter_2x -= 1

        result = (x[0].clip(0, 1) * 255).astype(np.uint8).transpose(1,2,0) # [3, H', W'] --> [H', W', 3]

        # alpha composition
        if alpha is not None:
            alpha = cv2.resize(alpha, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
            result = np.concatenate([result, np.expand_dims(alpha, -1)], axis=-1)

        # adjust for non-propotional scaling
        if result.shape[0] != out_size[0] or result.shape[1] != out_size[1]:
            result = cv2.resize(result, (out_size[1], out_size[0]), interpolation=cv2.INTER_CUBIC)

        # ocr text (experimental, not working well with the ultra light model...)
        if self.ocr_text:
            
            ocr_outputs = self.OCR.detect_and_ocr(result)
            result = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
            from . import ppocronnx
            result = ppocronnx.utility.draw_ocr_replace_txt(result, ocr_outputs, 
                font_size=self.ocr_font_size,
                font_color=self.ocr_font_color, 
                background_color=self.ocr_background_color,
                font_path=self.ocr_font_ttf,
                drop_score=0.3)

        return result
