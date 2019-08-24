from math import atan2, sqrt
from functools import partial, partialmethod
from itertools import starmap
from operator import and_, or_, xor, invert
from typing import Tuple, Optional

class Color(object):
    __slots__ = ('_color')
    _RGBfmt = frozenset({'RGB', 'float'})
    _HSLfmt = frozenset({'HSL', 'luminance'})
    _HSVfmt = frozenset({'HSV', 'value'})
    _HSIfmt = frozenset({'HSI', 'intensity'})
    _HCL709 = frozenset({'709', 'HCL709', 'Luma sRGB'})
    _HCL601 = frozenset({'601', 'HCL601', 'Luma NTSC'})
    _CMYK = frozenset({'CMYK', 'printer'})
    _XYZfmt = frozenset({'XYZ', 'CIE XYZ', 'D65 XYZ'})
    _LABfmt = frozenset({'Lab', 'CIE Lab'})
    formats = ('RGB', 'CMYK', 'HSL', 'HSV', 'HSI', 'HCL709', 'HCL601', 'CIE XYZ', 'CIE Lab')

    def __init__(self, r: int, g: int, b: int) -> None:
        r, g, b = map(lambda x: x & 0xFF, (r, g, b))
        self._color = r << 16 | g << 8 | b

    def __bool__(self) -> bool:
        return bool(self._color)

    def __str__(self) -> str:
        return hex(self._color)

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self.color}'

    def __getattr__(self, name: str):
        if name == 'r':
            return self._color >> 16
        if name == 'g':
            return self._color >> 8 & 0xFF
        if name == 'b':
            return self._color & 0xFF
        if name == 'color':
            return (
                self._color >> 16,
                self._color >> 8 & 0xFF,
                self._color & 0xFF
                )

    def __iter__(self):
        return iter(
            self._color >> 16,
            self._color >> 8 & 0xFF,
            self._color & 0xFF
            )

    @classmethod
    def from_hex(cls: type, code: str):
        color = object.__new__(cls)
        color._color = int(code, 16) & 0xFFFFFF
        return color

    @staticmethod
    def _hsl2rgbn(h, s, l, n):
        k = (n + 12*h) % 12
        return round(255*(l - s*min(l, 1-l)*max(min(k-3, 9-k, 1), -1)))

    @staticmethod
    def _hsv2rgbn(h, s, v, n):
        k = (n + 6*h) % 6
        return round(255*v*(1 - s*max(min(k, 4-k, 1), 0)))

    @staticmethod
    def _hsi2rgbn(h, s, i, n):
        k = (n + 6*h) % 6
        return round(255*(i*(1-s) + (3*s*i / (2-abs(6*h%2 - 1)))*max(min(k, 4-k, 1), 0)))

    @staticmethod
    def _hcl2rgbn(h, c, n):
        k = (n + 6*h) % 6
        return c*max(min(k, 4-k, 1), 0)

    @staticmethod
    def _xyzgminv(u):
        if u <= 0.0031308:
            v = 12.92*u
        else:
            v = 1.055*u**(5/12) - 0.055
        return round(255*min(max(v, 0), 1))

    @staticmethod
    def _xyzgamma(u):
        if u <= 0.04045: return u/12.92
        return ((u + 0.055)/1.055)**2.4

    @staticmethod
    def _labinv(t):
        d = 6/29
        if t > d: return t*t*t
        return 3*d*d*(t - 4/29)

    @staticmethod
    def _labfwd(t):
        d = 6/29
        if t > d*d*d: return t**(1/3)
        return t/3/d/d + 4/29

    @classmethod
    def from_format(cls, fmt: str, a: float, b: float, c: float, d: Optional[float]=None):
        if fmt in cls._XYZfmt:
            if d is not None:
                raise TypeError('expected 3 color arguments for format, got 4')
            if not(0 <= a <= 0.950489 and 0 <= b <= 1 and 0 <= c <= 1.08884):
                raise ValueError('color is outside standard gamut')
            r = 3.2406*a - 1.5372*b - 0.4986*c
            g = -0.9689*a + 1.8758*b + 0.0415*c
            b = 0.0557*a - 0.204*b + 1.057*c
            return cls(*map(cls._xyzgminv, (r, g, b)))
        if fmt in cls._LABfmt:
            if d is not None:
                raise TypeError('expected 3 color arguments for format, got 4')
            if not(0 <= a <= 100 and abs(b) <= 128 and abs(c) <= 128):
                raise ValueError('color is outside standard gamut')
            x = 0.950489*cls._labinv((a+16)/116 + b/500)
            y = cls._labinv((a+16)/116)
            z = 1.08884*cls._labinv((a+16)/116 - c/200)
            r = 3.2406*x - 1.5372*y - 0.4986*z
            g = -0.9689*x + 1.8758*y + 0.0415*z
            b = 0.0557*x - 0.204*y + 1.057*z
            return cls(*map(cls._xyzgminv, (r, g, b)))
        if any(not 0. <= v <= 1. for v in (a, b, c)):
            raise ValueError('color values must be in a range from 0.0 to 1.0')
        if fmt in cls._CMYK:
            if d is None:
                raise TypeError('expected 4 color arguments for format, got 3')
            r, g, b = (1-a)*(1-d), (1-b)*(1-d), (1-c)*(1-d)
            return cls(*map(lambda a: round(255*a), (r, g, b)))
        elif d is not None:
            raise TypeError('expected 3 color arguments for format, got 4')
        if fmt in cls._RGBfmt:
            return cls(*map(lambda a: round(255*a), (a, b, c)))
        if fmt in cls._HSLfmt:
            if b == 0: return cls(*map(round, (c, c, c)))
            return cls(*map(partial(cls._hsl2rgbn, a, b, c), (0, 8, 4)))
        if fmt in cls._HSVfmt:
            if b == 0: return cls(*map(round, (c, c, c)))
            return cls(*map(partial(cls._hsv2rgbn, a, b, c), (5, 3, 1)))
        if fmt in cls._HSIfmt:
            if b == 0: return cls(*map(round, (c, c, c)))
            return cls(*map(partial(cls._hsi2rgbn, a, b, c), (2, 0, 4)))
        if fmt in cls._HCL709:
            r, g, b = map(partial(cls._hcl2rgbn, a, b), (2, 0, 4))
            m = c - (0.2126*r + 0.7152*g + 0.0722*b)
            return cls(*map(lambda a: round(255*a), (r+m, g+m, b+m)))
        if fmt in cls._HCL601:
            r, g, b = map(partial(cls._hcl2rgbn, a, b), (2, 0, 4))
            m = c - (0.2989*r + 0.5870*g + 0.1140*b)
            return cls(*map(lambda a: round(255*a), (r+m, g+m, b+m)))
        raise ValueError('color format name invalid')

    def to_format(self, fmt: str) -> Tuple[float, ...]:
        r = self._color >> 16
        g = self._color >> 8 & 0xFF
        b = self._color & 0xFF
        if fmt in self._RGBfmt:
            return tuple(r/255, g/255, b/255)
        if fmt in self._XYZfmt:
            r, g, b = map(lambda x: self._xyzgamma(x/255), (r, g, b))
            x = 0.4124*r + 0.3576*g + 0.1805*b
            y = 0.2126*r + 0.7152*g + 0.0722*b
            z = 0.0193*r + 0.1192*g + 0.9505*b
            return (x, y, z)
        if fmt in self._LABfmt:
            r, g, b = map(lambda x: self._xyzgamma(x/255), (r, g, b))
            x = 0.4124*r + 0.3576*g + 0.1805*b
            y = 0.2126*r + 0.7152*g + 0.0722*b
            z = 0.0193*r + 0.1192*g + 0.9505*b
            l = 116*self._labfwd(y) - 16
            a = 500*(self._labfwd(x/.950489) - self._labfwd(y))
            b = 200*(self._labfwd(y) - self._labfwd(z/1.08884))
            return (l, a, b)
        mx, mn = max(r, g, b), min(r, g, b)
        if fmt in self._CMYK:
            k = 255 - mx
            return (*map(lambda x: (255-x-k) / mx, (r, g, b)), k / 255)
        chroma = mx - mn
        if chroma == 0: hue = 0.
        elif mx == r: hue = (g-b) / chroma % 6
        elif mx == g: hue = 2 + (b-r) / chroma
        elif mx == b: hue = 4 + (r-g) / chroma
        hue /= 6
        if fmt in self._HSLfmt:
            sat = 0 if mx == 0 else chroma / (255 - abs(mx+mn-255))
            return (hue, sat, (mx+mn) / 510)
        if fmt in self._HSVfmt:
            sat = 0 if mx == 0 or mn == 255 else chroma / mx
            return (hue, sat, mx / 255)
        if fmt in self._HSIfmt:
            its = (r + g + b) / 3
            return (hue, 1 - mn / its, its / 255)
        if fmt in self._HCL709:
            lum = (0.2126*r + 0.7152*g + 0.0722*b)
            return (hue, chroma / 255, lum / 255)
        if fmt in self._HCL601:
            lum = (0.2989*r + 0.5870*g + 0.1140*b)
            return (hue, chroma / 255, lum / 255)
        raise ValueError('color format name invalid')

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)(
            *starmap(lambda x: min(max(x + y, 0), 1)),
            *zip(self.color, other.color),
            )

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)(
            *starmap(lambda x: min(max(x - y, 0), 1)),
            *zip(self.color, other.color),
            )

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)(
            *starmap(lambda x: min(max(1-(1-x)*(1-y), 0), 1)),
            *zip(self.color, other.color),
            )

    def _boolop(self, op, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)(*starmap(op, *zip(self.color, other.color)))

    __or__ = partialmethod(_boolop, or_)
    __and__ = partialmethod(_boolop, and_)
    __xor__ = partialmethod(_boolop, xor_)

    def __invert__(self):
        return type(self)(*map(invert, zip(self.color)))
