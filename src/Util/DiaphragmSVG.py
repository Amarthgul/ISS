
from xml.etree import ElementTree as ET
from copy import deepcopy
from typing import Optional, Tuple, Iterable
from svgpathtools import parse_path
from PIL import Image, ImageDraw, ImageColor
import math
import matplotlib.pyplot as plt


from .Backend import backend as bd
from .Misc import RectPath


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _ns(tag: str) -> str:
    """Attach SVG namespace to a tag if it doesn't have one."""
    if tag.startswith("{"):
        return tag
    return f"{{{SVG_NS}}}{tag}"


def _iter_all_elements(elem: ET.Element) -> Iterable[ET.Element]:
    """Yield element and all descendants."""
    yield elem
    for e in elem:
        yield from _iter_all_elements(e)


def _get_attr_float(el: ET.Element, key: str) -> Optional[float]:
    v = el.attrib.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _find_point_coords(root: ET.Element, point_id: str) -> Tuple[float, float]:
    """
    Find a 'point' by id and return (x,y).
    Accepts:
      * elements with (cx,cy) or (x,y)
      * <path> whose first command is M/m (uses that coordinate)
    Searches the element first, then its descendants.
    """
    target = root.find(f".//*[@id='{point_id}']")
    if target is None:
        raise ValueError(f"Element with id='{point_id}' not found.")

    def _coords_on(el: ET.Element) -> Optional[Tuple[float, float]]:
        cx = el.attrib.get("cx"); cy = el.attrib.get("cy")
        if cx is not None and cy is not None:
            try:
                return (float(cx), float(cy))
            except ValueError:
                pass

        x = el.attrib.get("x"); y = el.attrib.get("y")
        if x is not None and y is not None:
            try:
                return (float(x), float(y))
            except ValueError:
                pass

        # Path case: try to read first moveto from 'd'
        d = el.attrib.get("d")
        if d:
            # very light-weight parse: look for first M or m followed by a pair of numbers
            import re
            m = re.search(r"[Mm]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[, ]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", d)
            if m:
                try:
                    return (float(m.group(1)), float(m.group(2)))
                except ValueError:
                    pass
        return None

    # check target itself
    coords = _coords_on(target)
    if coords is not None:
        return coords

    # then any descendant
    for child in _iter_all_elements(target):
        coords = _coords_on(child)
        if coords is not None:
            return coords

    raise ValueError(f"Could not find (cx,cy) or (x,y) within id='{point_id}', "
                     f"and no usable moveto in a path.")


def _append_class(el: ET.Element, cls: str) -> None:
    prev = el.attrib.get("class", "")
    parts = [p for p in prev.split() if p]
    if cls not in parts:
        parts.append(cls)
    if parts:
        el.set("class", " ".join(parts))


def _append_transform(el: ET.Element, t: str) -> None:
    prev = el.attrib.get("transform")
    if prev:
        el.set("transform", prev + " " + t)
    else:
        el.set("transform", t)


def _parse_style_attr(style_str: str) -> dict:
    # "fill:#abc; fill-opacity:0.5; opacity:0.8"
    out = {}
    if not style_str:
        return out
    for decl in style_str.split(";"):
        if ":" in decl:
            k, v = decl.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _compute_local_fill(el, inherited_rgba):
    """
    Return the effective RGBA fill for this element, falling back to inherited_rgba.
    Supports: fill, fill-opacity, opacity, and style="" overrides.
    If fill is 'none', returns None (skip drawing).
    """
    # start from inherited
    rgba = inherited_rgba

    # gather inline style + attributes
    style = _parse_style_attr(el.get("style"))
    fill_val = el.get("fill", None)
    if "fill" in style:
        fill_val = style["fill"]

    # handle 'none'
    if fill_val is not None and fill_val.lower() == "none":
        return None

    # choose base color if provided locally
    if fill_val is not None:
        # ImageColor.getrgb handles #rgb, #rrggbb, rgb(), named colors, etc.
        r, g, b = ImageColor.getrgb(fill_val)
    else:
        if rgba is not None:
            r, g, b, _ = rgba
        else:
            r, g, b = (0, 0, 0)  # default if nothing inherited

    # alpha from fill-opacity or (fallback) opacity; multiply with inherited alpha
    alpha = 255
    if rgba is not None:
        alpha = rgba[3]

    fill_opacity = el.get("fill-opacity", None)
    if "fill-opacity" in style:
        fill_opacity = style["fill-opacity"]

    opacity = el.get("opacity", None)
    if "opacity" in style:
        opacity = style["opacity"]

    a_scale = 1.0
    if fill_opacity is not None:
        a_scale *= float(fill_opacity)
    if opacity is not None:
        a_scale *= float(opacity)

    a = int(round((alpha / 255.0) * a_scale * 255.0))
    return (r, g, b, a)


def _ensure_group_wrapper(el: ET.Element) -> ET.Element:
    """
    Ensure an element is inside a <g>. If 'el' is already <g>, return it.
    Otherwise, replace el with a <g> wrapper having same id (moved to wrapper)
    and put el inside.
    """
    if el.tag.endswith("}g"):
        return el

    parent = el.getparent() if hasattr(el, "getparent") else None  # for lxml, but ElementTree lacks getparent
    # ElementTree doesn't provide getparent natively; so we handle grouping
    # in a simpler way: we create a <g> in the same parent by scanning.
    # But since stock ElementTree lacks parent links, we will avoid this
    # utility and instead create <g> wrappers only for duplicates where we
    # fully control insertion. For originals, we'll mark them directly.
    return el


def _parse_transform(transform_str: str):
    # returns 3x3 affine matrix as tuple-of-tuples
    # supports: matrix(a b c d e f), translate(tx[,ty]), scale(sx[,sy]), rotate(angle[, cx, cy])
    def matmul(A, B):
        return (
            (A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
             A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
             A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]),
            (A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
             A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
             A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]),
            (A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
             A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
             A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]),
        )

    I = ((1,0,0),(0,1,0),(0,0,1))
    if not transform_str:
        return I

    import re
    tokens = re.findall(r'(matrix|translate|scale|rotate)\s*\(([^)]*)\)', transform_str)
    M = I
    for kind, inside in tokens:
        nums = [float(v) for v in re.split(r'[,\s]+', inside.strip()) if v]
        if kind == 'matrix':
            a,b,c,d,e,f = nums
            T = ((a,c,e),(b,d,f),(0,0,1))
        elif kind == 'translate':
            tx = nums[0]; ty = nums[1] if len(nums)>1 else 0.0
            T = ((1,0,tx),(0,1,ty),(0,0,1))
        elif kind == 'scale':
            sx = nums[0]; sy = nums[1] if len(nums)>1 else sx
            T = ((sx,0,0),(0,sy,0),(0,0,1))
        elif kind == 'rotate':
            ang = math.radians(nums[0])
            ca, sa = math.cos(ang), math.sin(ang)
            if len(nums) == 3:
                cx, cy = nums[1], nums[2]
                # translate(-cx,-cy) * R * translate(cx,cy)
                T = matmul(
                    matmul(((1,0,cx),(0,1,cy),(0,0,1)),
                           ((ca,-sa,0),(sa,ca,0),(0,0,1))),
                    ((1,0,-cx),(0,1,-cy),(0,0,1))
                )
            else:
                T = ((ca,-sa,0),(sa,ca,0),(0,0,1))
        M = matmul(M, T)
    return M


def _apply_affine(M, x, y):
    return (M[0][0]*x + M[0][1]*y + M[0][2],
            M[1][0]*x + M[1][1]*y + M[1][2])


def _circle_poly(cx, cy, r, n=128):
    return [(cx + r*math.cos(2*math.pi*i/n), cy + r*math.sin(2*math.pi*i/n)) for i in range(n)]


def _ellipse_poly(cx, cy, rx, ry, n=128):
    return [(cx + rx*math.cos(2*math.pi*i/n), cy + ry*math.sin(2*math.pi*i/n)) for i in range(n)]


def _path_to_poly(d: str, samples_per_unit=1.0, min_samples=8, max_samples=256):
    # approximate any path into a polyline list (assumes closed shapes if Z present)
    P = parse_path(d)
    pts = []
    for seg in P:
        L = seg.length(error=1e-4)
        k = max(min_samples, min(max_samples, int(L * samples_per_unit)))
        for i in range(k):
            t = i / float(k)
            z = seg.point(t)
            pts.append((z.real, z.imag))
    if len(P) > 0:
        z = P[-1].point(1.0)
        pts.append((z.real, z.imag))
    return pts


class DiaphragmBlades:
    def __init__(self, svg_path:str):
        self.tree=ET.parse(svg_path)
        self.root=self.tree.getroot()
        if not self.root.tag.endswith("svg"):
            raise ValueError("Not an SVG root")


    def DuplicateAroundCenter(self,n:int,step:float,
                                               main_id="main",pivot_id="pivot",
                                               center_id="center",layer_id="generated_copies"):
        if n<1: return
        r=self.root
        main=r.find(f".//*[@id='{main_id}']"); pivot=r.find(f".//*[@id='{pivot_id}']")
        if main is None or pivot is None: raise ValueError("main or pivot not found")
        cx,cy=_find_point_coords(r,center_id)
        _append_class(main,"rot_target"); _append_class(pivot,"rot_target")
        layer=r.find(f".//*[@id='{layer_id}']")
        if layer is None:
            layer=ET.Element(_ns("g"),{"id":layer_id}); r.append(layer)
        for i in range(1,n+1):
            g=ET.Element(_ns("g"),{"id":f"pair_{i}"})
            mc,pc=deepcopy(main),deepcopy(pivot)
            mc.set("id",f"{main_id}_copy_{i}"); pc.set("id",f"{pivot_id}_copy_{i}")
            _append_class(mc,"rot_target"); _append_class(pc,"rot_target")
            g.extend([mc,pc])
            _append_transform(g,f"rotate({i*step} {cx} {cy})")
            layer.append(g)


    def RotateAllBlades(self,deg:float,pivot_id="pivot"):
        px,py=_find_point_coords(self.root,pivot_id)
        for e in self.root.iter():
            if "rot_target" in e.get("class","").split():
                _append_transform(e,f"rotate({deg} {px} {py})")


    def toArray(self) -> bd.ndarray:
        """
        Rasterize the current SVG (simple fills) to an RGBA uint8 array using svgpathtools for geometry
        and Pillow for fast polygon filling. Assumes simple shapes/paths; no gradients/filters.
        """
        # ------------- derive canvas from viewBox or width/height ---
        vb = self.root.get("viewBox")
        if vb:
            vx, vy, vw, vh = [float(x) for x in vb.replace(",", " ").split()]
            W = int(round(vw))
            H = int(round(vh))
        else:
            W = int(float(self.root.get("width", "512")))
            H = int(float(self.root.get("height", "512")))
            vx, vy, vw, vh = 0.0, 0.0, float(W), float(H)

        sx = W / vw
        sy = H / vh

        # ----------------- PIL image target (transparent BG) ---
        # 0-255
        img = Image.new("RGBA", (W, H), (0,0,0,0))
        drw = ImageDraw.Draw(img)

        # ------------ recursive traversal to accumulate transforms ---
        def render_elem(el, M_parent, inherited_fill_rgba):
            # accumulate own transform
            M_local = _parse_transform(el.get("transform"))
            # M_total = M_parent * M_local
            # (matrix multiply defined in helper uses A*B order)
            def matmul(A,B):
                return (
                    (A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
                     A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
                     A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2]),
                    (A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
                     A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
                     A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2]),
                    (A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
                     A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
                     A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2]),
                )
            M = matmul(M_parent, M_local)

            tag = el.tag.split('}',1)[-1]

            # --------- polygons from basic shapes ---
            poly = None

            if tag == "rect":
                x = float(el.get("x", "0")); y = float(el.get("y", "0"))
                w = float(el.get("width")); h = float(el.get("height"))
                rect = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
                poly = [ _apply_affine(M, X,Y) for (X,Y) in rect ]

            elif tag == "circle":
                cx = float(el.get("cx")); cy = float(el.get("cy")); r = float(el.get("r"))
                circ = _circle_poly(cx, cy, r, n=128)
                poly = [ _apply_affine(M, X,Y) for (X,Y) in circ ]

            elif tag == "ellipse":
                cx = float(el.get("cx")); cy = float(el.get("cy"))
                rx = float(el.get("rx")); ry = float(el.get("ry"))
                ell = _ellipse_poly(cx, cy, rx, ry, n=128)
                poly = [ _apply_affine(M, X,Y) for (X,Y) in ell ]

            elif tag == "polygon":
                pts = el.get("points").replace(",", " ").split()
                it = iter(pts); pg = [ (float(a), float(b)) for a,b in zip(it,it) ]
                poly = [ _apply_affine(M, X,Y) for (X,Y) in pg ]

            elif tag == "path":
                d = el.get("d")
                path_poly = _path_to_poly(d, samples_per_unit=1.0, min_samples=8, max_samples=256)
                poly = [ _apply_affine(M, X,Y) for (X,Y) in path_poly ]

            # compute this element's effective fill (inherits from parent)
            current_fill_rgba = _compute_local_fill(el, inherited_fill_rgba)

            # --- rasterize polygon if present ---
            if poly and len(poly) >= 3:
                # map from SVG coords to pixel coords via viewBox scaling
                screen = [ ((X - vx)*sx, (Y - vy)*sy) for (X,Y) in poly ]
                # simple fill (black, fully opaque)
                drw.polygon(screen, fill=(0,0,0,255))

            # recurse into children
            for child in el:
                render_elem(child, M, current_fill_rgba)

        # start traversal
        I = ((1,0,0),(0,1,0),(0,0,1))
        default_fill = (0, 0, 0, 255)
        render_elem(self.root, I, default_fill)



        arr = bd.array(img, dtype=bd.uint8)  # (H, W, 4)

        arr = self._MultiplyCircle(arr)

        return bd.asarray(arr)


    def DrawDiaphragm(self):

        pass


    def _MultiplyCircle(self, img: bd.ndarray) -> bd.ndarray:
        """
        Given a square RGBA image array (H, W, 4), create a black image of the
        same size, overlay a white circle (diameter = image size), and multiply
        the two images elementwise for RGB channels. The alpha channel is
        additive (logical OR): resulting alpha = 1 - (1 - a_img)*(1 - a_circle).

        Returns a uint8 RGBA array.
        """
        h, w, c = img.shape
        assert c == 4 and h == w, "Input must be a square RGBA image (H==W, 4 channels)."
        n = h

        # 1) Generate white circle mask (same size)
        yy, xx = bd.ogrid[:n, :n]
        cx = (n - 1) / 2.0
        cy = (n - 1) / 2.0
        r = n / 2.0
        circle_mask = (xx - cx)**2 + (yy - cy)**2 <= r**2

        # 2) Create the white circle RGBA image
        circle_img = bd.zeros_like(img, dtype=bd.uint8)
        circle_img[..., 0:3][circle_mask] = 255
        circle_img[..., 3][~circle_mask] = 255

        # 3) Multiply RGB channels (normalized)
        result_rgb = (img[..., :3].astype(bd.uint16) * circle_img[..., :3].astype(bd.uint16)) // 255

        # 4) Combine alpha channels additively (OR)
        a_img = img[..., 3].astype(bd.float32) / 255.0
        a_circle = circle_img[..., 3].astype(bd.float32) / 255.0
        # logical OR in continuous form: a_out = 1 - (1 - a1)*(1 - a2)
        a_out = 1.0 - (1.0 - a_img) * (1.0 - a_circle)
        result_alpha = (a_out * 255.0).astype(bd.uint8)

        # 5) Stack final RGBA
        result = bd.zeros_like(img, dtype=bd.uint8)
        result[..., :3] = result_rgb
        result[..., 3] = result_alpha

        return result
