"""Trusted fixed image identity for visual acceptance, not a user fault selector."""
import hashlib
from io import BytesIO
import re
from pathlib import Path


def load_fixed_image(path, expected_sha256=None):
    from PIL import Image
    if expected_sha256 is not None and (not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256)):
        raise ValueError('FULL_IMAGE_SHA256_REQUIRED')
    raw=Path(path).read_bytes()
    actual=hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and actual!=expected_sha256:
        raise ValueError('FIXED_IMAGE_SHA256_MISMATCH')
    with Image.open(BytesIO(raw)) as opened:
        if opened.size!=(64,64):raise ValueError('FIXED_IMAGE_64X64_REQUIRED')
        image=opened.convert('RGB')
    return image,dict(source='DEPLOYMENT_FIXED_FILE',file_sha256=actual,
        expected_sha256_check=expected_sha256 is not None,
        bytes=len(raw),size=[64,64],pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
