import contextlib


class QuietFallback:
    def __getattr__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self


try:
    with contextlib.redirect_stdout(None):
        import pygame
except ModuleNotFoundError:
    pygame = QuietFallback()

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, *args, **kwargs: iterable
