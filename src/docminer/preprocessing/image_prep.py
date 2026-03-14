"""Image preprocessing utilities for OCR improvement."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def deskew(image: "PILImage.Image") -> "PILImage.Image":
    """Detect and correct page skew using Hough-transform line angle estimation.

    Falls back to the original image if numpy/PIL are unavailable or if
    no dominant skew angle is detected.

    Parameters
    ----------
    image:
        Input PIL image (RGB or grayscale).

    Returns
    -------
    PIL Image
        Deskewed (rotated) image.
    """
    try:
        import numpy as np
        from PIL import Image

        img_gray = image.convert("L")
        arr = np.array(img_gray)

        # Binarise for edge detection
        threshold = arr.mean()
        binary = (arr < threshold).astype(np.uint8) * 255

        # Estimate skew angle using projection profile variance
        angle = _estimate_skew_angle(binary)

        if abs(angle) < 0.5:
            return image  # No significant skew

        logger.debug("Deskewing by %.2f degrees", angle)
        return image.rotate(angle, expand=True, fillcolor=255)

    except ImportError:
        logger.debug("numpy not available; skipping deskew")
        return image
    except Exception as exc:
        logger.debug("Deskew failed: %s", exc)
        return image


def _estimate_skew_angle(binary_arr) -> float:
    """Estimate skew angle from a binary image array.

    Uses horizontal projection profile: finds the rotation angle that
    maximises variance of row sums (i.e., text lines become aligned).
    """
    import numpy as np

    best_angle = 0.0
    best_variance = 0.0

    for angle_deg in range(-15, 16, 1):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        h, w = binary_arr.shape
        # Rotate coordinates
        cy, cx = h / 2, w / 2
        coords = np.array(
            [(int(cx + (x - cx) * cos_a - (y - cy) * sin_a),
              int(cy + (x - cx) * sin_a + (y - cy) * cos_a))
             for y in range(0, h, 4)
             for x in range(0, w, 4)]
        )
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < w) &
            (coords[:, 1] >= 0) & (coords[:, 1] < h)
        )
        valid_coords = coords[valid]
        if len(valid_coords) == 0:
            continue

        rotated_proj = np.zeros(h)
        for px, py in valid_coords:
            rotated_proj[py] += binary_arr[py, px]

        variance = float(np.var(rotated_proj))
        if variance > best_variance:
            best_variance = variance
            best_angle = angle_deg

    return best_angle


def denoise(image: "PILImage.Image") -> "PILImage.Image":
    """Apply bilateral filter denoising to reduce scanner noise.

    Falls back gracefully when OpenCV is not available.

    Parameters
    ----------
    image:
        Input PIL image.

    Returns
    -------
    PIL Image
        Denoised image.
    """
    try:
        import cv2
        import numpy as np

        arr = np.array(image.convert("RGB"))
        # Bilateral filter: preserves edges while smoothing noise
        denoised = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
        from PIL import Image

        return Image.fromarray(denoised)
    except ImportError:
        logger.debug("OpenCV not available; skipping denoise (using PIL filter)")
        # Fallback: PIL median filter
        try:
            from PIL import ImageFilter

            return image.filter(ImageFilter.MedianFilter(size=3))
        except Exception:
            return image
    except Exception as exc:
        logger.debug("Denoise failed: %s", exc)
        return image


def enhance_contrast(image: "PILImage.Image") -> "PILImage.Image":
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Falls back to simple histogram equalisation if OpenCV is unavailable.

    Parameters
    ----------
    image:
        Input PIL image.

    Returns
    -------
    PIL Image
        Contrast-enhanced image.
    """
    try:
        import cv2
        import numpy as np

        arr = np.array(image.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(arr)
        # Convert back to RGB
        from PIL import Image

        return Image.fromarray(enhanced).convert("RGB")
    except ImportError:
        logger.debug("OpenCV not available; using PIL contrast enhancement")
        try:
            from PIL import ImageOps

            gray = image.convert("L")
            equalized = ImageOps.equalize(gray)
            return equalized.convert("RGB")
        except Exception:
            return image
    except Exception as exc:
        logger.debug("Contrast enhancement failed: %s", exc)
        return image


def binarize(image: "PILImage.Image") -> "PILImage.Image":
    """Apply adaptive threshold binarisation for clean B/W OCR input.

    Uses Gaussian adaptive thresholding (OpenCV) when available,
    otherwise falls back to Otsu's method via PIL.

    Parameters
    ----------
    image:
        Input PIL image (any mode).

    Returns
    -------
    PIL Image
        Binary (black-and-white) image.
    """
    try:
        import cv2
        import numpy as np

        arr = np.array(image.convert("L"))
        binary = cv2.adaptiveThreshold(
            arr,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        from PIL import Image

        return Image.fromarray(binary).convert("RGB")
    except ImportError:
        logger.debug("OpenCV not available; using Otsu binarization via PIL")
        try:
            import numpy as np
            from PIL import Image

            gray = np.array(image.convert("L"))
            # Otsu threshold
            threshold = _otsu_threshold(gray)
            binary = (gray > threshold).astype(np.uint8) * 255
            return Image.fromarray(binary).convert("RGB")
        except ImportError:
            # Pure PIL fallback
            from PIL import Image

            gray = image.convert("L")
            return gray.point(lambda p: 255 if p > 128 else 0).convert("RGB")
    except Exception as exc:
        logger.debug("Binarization failed: %s", exc)
        return image


def _otsu_threshold(arr) -> int:
    """Compute Otsu's binarization threshold from a grayscale numpy array."""
    import numpy as np

    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0
    max_var = 0.0
    threshold = 128

    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        var_between = float(w_b) * float(w_f) * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold
