
from xml.etree import ElementTree as ET
from copy import deepcopy
from typing import Optional, Tuple, Iterable

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


# ------------------------
# Core class
# ------------------------

class DiaphragmBlades:
    def __init__(self, svg_path: str):
        self.svg_path = svg_path
        self.tree = ET.parse(svg_path)
        self.root = self.tree.getroot()
        if not self.root.tag.endswith("svg"):
            raise ValueError("Root is not an <svg> element.")


    def save(self, out_path: str) -> None:
        self.tree.write(out_path, encoding="utf-8", xml_declaration=True)


    def Show(self, use_browser: bool = False):
        """
        Display the SVG content.
        - If use_browser=True, open it in the default web browser.
        - Otherwise, print the XML text (if in notebook, it will render inline).
        """
        import io, webbrowser, tempfile, sys
        svg_str = ET.tostring(self.root, encoding="unicode")

        if use_browser:
            # Write to a temp file and open
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".svg", mode="w", encoding="utf-8")
            tmp.write(svg_str)
            tmp.close()
            webbrowser.open(f"file://{tmp.name}")
            print(f"Opened in browser: {tmp.name}")
        else:
            # If running in IPython/Jupyter, display inline
            if "ipykernel" in sys.modules:
                from IPython.display import SVG, display
                display(SVG(svg_str))
            else:
                print(svg_str)


    # 1) Duplicate "main" and "pivot", rotate them around "center"
    def DuplicateAroundCenter(self, n_copies: int, step_deg: float,
                              main_id: str = "main",
                              pivot_id: str = "pivot",
                              center_id: str = "center",
                              layer_id: str = "generated_copies") -> None:
        """
        Create n_copies duplicates (not counting the original) of 'main' and 'pivot'.
        For copy i (1..n_copies), rotate by i*step_deg around 'center'. Insert into a wrapper <g id=layer_id>.
        Mark all originals & duplicates that should rotate later with class='rot_target'.
        """
        if n_copies < 1:
            return

        # Find elements
        main = self.root.find(f".//*[@id='{main_id}']")
        if main is None:
            raise ValueError(f"Element with id='{main_id}' not found.")
        pivot = self.root.find(f".//*[@id='{pivot_id}']")
        if pivot is None:
            raise ValueError(f"Element with id='{pivot_id}' not found.")

        cx, cy = _find_point_coords(self.root, center_id)

        # Mark originals as rotation targets too
        _append_class(main, "rot_target")
        _append_class(pivot, "rot_target")

        # Make (or find) a layer to hold duplicates
        layer = self.root.find(f".//*[@id='{layer_id}']")
        if layer is None:
            layer = ET.Element(_ns("g"), {"id": layer_id})
            self.root.append(layer)

        for i in range(1, n_copies + 1):
            theta = i * step_deg

            main_copy = deepcopy(main)
            pivot_copy = deepcopy(pivot)

            # Ensure unique ids on copies if the originals had ids (avoid collisions)
            if "id" in main_copy.attrib:
                main_copy.set("id", f"{main_id}_copy_{i}")
            if "id" in pivot_copy.attrib:
                pivot_copy.set("id", f"{pivot_id}_copy_{i}")

            # Put them inside a group so the rotation applies to both together
            group = ET.Element(_ns("g"), {"id": f"pair_{i}"})
            group.append(main_copy)
            group.append(pivot_copy)

            # Mark as rotation targets
            _append_class(main_copy, "rot_target")
            _append_class(pivot_copy, "rot_target")

            # Rotate group around center by theta
            _append_transform(group, f"rotate({theta} {cx} {cy})")

            layer.append(group)


    def RotateAllBlades(self, angle_deg: float, pivot_id: str = "pivot") -> None:
        """
        Append a rotation transform around the (global) 'pivot' coordinates to every element
        marked with class='rot_target' (including duplicates).
        """
        px, py = _find_point_coords(self.root, pivot_id)

        # Find all rotatable elements (any element with class containing 'rot_target')
        # We'll use an XPath-like contains() by scanning all elements.
        for el in self.root.iter():
            cls = el.attrib.get("class", "")
            if "rot_target" in cls.split():
                # Append rotation around pivot point
                prev = el.attrib.get("transform")
                rotate = f"rotate({angle_deg} {px} {py})"
                if prev:
                    el.set("transform", prev + " " + rotate)
                else:
                    el.set("transform", rotate)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Duplicate and rotate elements in an SVG.")
    p.add_argument("input_svg", help="Path to input SVG")
    p.add_argument("output_svg", help="Path to write the modified SVG")
    p.add_argument("--copies", type=int, default=7, help="Number of duplicate pairs to create")
    p.add_argument("--step", type=float, default=30.0, help="Degrees between duplicates around center")
    p.add_argument("--spin", type=float, default=0.0, help="Extra rotation for all rot_target around pivot")
    p.add_argument("--main", default="main", help="ID of the main shape")
    p.add_argument("--pivot", default="pivot", help="ID of the pivot point element")
    p.add_argument("--center", default="center", help="ID of the center point element")
    p.add_argument("--layer", default="generated_copies", help="ID for the wrapper layer for duplicates")
    args = p.parse_args()

    rot = DiaphragmBlades(args.input_svg)
    rot.DuplicateAroundCenter(args.copies, args.step,
                              main_id=args.main,
                              pivot_id=args.pivot,
                              center_id=args.center,
                              layer_id=args.layer)
    if args.spin != 0.0:
        rot.RotateAllBlades(args.spin, pivot_id=args.pivot)
    rot.save(args.output_svg)
