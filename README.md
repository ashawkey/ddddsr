# ddddsr

A python library for end-to-end image super-resolution.

### Install

```bash
pip install ddddsr
```

### Features

* End-to-end super resolution in one package (currently support `waifu2x` models).

* (experimental) OCR for better text super resolution quality.

### Usage

```python
import ddddsr

# read input.jpg and write to output.png
# default is 2x upscaling on CPU using waifu2x cunet.
ddddsr.SR()('input.jpg', 'output.png')
```

API reference:

```python
sr = ddddsr.SR( 
    # models: currently supports ['waifu2x_art', 'waifu2x_photo']
    model: str = 'waifu2x_art', 
    # scale
    scale: Optional[float] = None, 
    # output size (omitted if scale is set, if both scale and size are None, use scale = 2)
    size: Optional[Union[int, List[int], Tuple[int]]] = None, 
    # denoise level: range in [-1, 3], -1 means no denoising.
    denoise_level: int =  2, 
    # whether to use gpu
    use_gpu: bool = False, 
    # if use gpu, set the device id
    device_id: int = 0,
    # (experimental) use OCR for better text quality
    ocr_text: bool = False,
    # (experimental) OCR related configurations
    ocr_font_size: int = 28,
    ocr_font_color: Union[Tuple[int], List[int]] = (0, 0, 0),
    ocr_background_color: Union[Tuple[int], List[int]] = (255, 255, 255),
    ocr_font_ttf: Optional[str] = None, # path for ttf font file
    # verbose 
    verbose: bool = False,
)

sr(
    # image, support array of [H, W, 3] or [H, W, 4], or the file path.
    image: Union[np.ndarray, str],
    # output_path, output file path, if is None, will return the ndarray.
    output_path: str = None,
    # slide window size, -1 means no slide window.
    window: int = 256,
)
```



### References
* The original [waifu2x](https://github.com/nagadomi/waifu2x).

* ONNX models are from [waifu2x-onnx](https://github.com/tcyrus/waifu2x-onnx).

* OCR models are from [ppocr-onnx](https://github.com/triwinds/ppocr-onnx) and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

  