"""API storage lives beside the backend, independently of the model root."""
import os
from pathlib import Path

from fashion.config import ROOT as CORE_ROOT

BE_ROOT = CORE_ROOT.parent / 'be'
UPLOAD_DIR = Path(os.environ.get('FASHION_UPLOAD_DIR', BE_ROOT / 'tmp/demo-api/uploads'))
