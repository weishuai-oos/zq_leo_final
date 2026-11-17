"""Create a separate missing-head asset without modifying the source robot."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


HEAD_LINK_NAMES = frozenset(("LINK_HEAD_PITCH", "LINK_HEAD_YAW"))
HEAD_JOINT_NAMES = frozenset(("J27_HEAD_PITCH", "J28_HEAD_YAW"))


def make_headless_urdf(source_urdf: str | Path, destination: str | Path | None = None) -> str:
    """Remove both head links and their joints; retain absolute mesh references.

    The result is intended for a separate evaluation scene. A passive attached
    head and a head hanging from wires are mechanically different scenarios.
    """
    source = Path(source_urdf).expanduser().resolve(strict=True)
    source_bytes = source.read_bytes()
    root = ET.fromstring(source_bytes)
    links = {element.attrib["name"] for element in root.findall("link")}
    if not HEAD_LINK_NAMES.issubset(links):
        raise ValueError(f"Source URDF does not contain the two expected head links: {source}")
    removed = set(HEAD_LINK_NAMES)
    # Also remove any attached descendants to avoid dangling joints if an
    # evaluation URDF later adds fixed visual links beneath the head.
    changed = True
    while changed:
        changed = False
        for joint in root.findall("joint"):
            parent, child = joint.find("parent"), joint.find("child")
            if parent is not None and child is not None and parent.attrib["link"] in removed:
                if child.attrib["link"] not in removed:
                    removed.add(child.attrib["link"])
                    changed = True
    for element in list(root):
        if element.tag == "link" and element.attrib.get("name") in removed:
            root.remove(element)
        elif element.tag == "joint":
            referenced = {part.attrib.get("link") for tag in ("parent", "child") for part in element.findall(tag)}
            if element.attrib.get("name") in HEAD_JOINT_NAMES or referenced.intersection(removed):
                root.remove(element)
        elif element.tag in ("transmission", "gazebo"):
            references = {part.attrib.get("name") for part in element.iter("joint")}
            if references.intersection(HEAD_JOINT_NAMES) or element.attrib.get("reference") in removed:
                root.remove(element)
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename and "://" not in filename and not Path(filename).is_absolute():
            mesh.attrib["filename"] = str((source.parent / filename).resolve())
    if destination is None:
        digest = hashlib.sha256(str(source).encode() + source_bytes).hexdigest()[:20]
        destination = Path(tempfile.gettempdir()) / "t800_impact_assets" / digest / "headless.urdf"
    target = Path(destination).expanduser().resolve()
    if target == source:
        raise ValueError("Headless URDF destination must differ from the source.")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    # Concurrent ranks may request the same generated asset. Atomic replacement
    # prevents an importer from seeing a partially written document.
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".urdf", delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(payload)
            output.close()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return str(target)
