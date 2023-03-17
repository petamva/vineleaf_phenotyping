from types import SimpleNamespace
from PIL import ImageFont
# FACTOR = 0.715

cfg = SimpleNamespace(
    phenot = SimpleNamespace(
        path='<path to the model>',
        img_size=(512, 512),
        f1 = 2.54/(512*600),
        f2 = (2.54/(512*600))**2,
        font=ImageFont.truetype('./fonts/SourceCodePro-Regular.ttf', 16),
    ),
)