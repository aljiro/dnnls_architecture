"""Parsers for the StoryReasoning annotations: the grounded story markup and the chain-of-thought tables."""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


def parse_gdi_text(text: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(text, "html.parser")
    result = []
    for index, gdi in enumerate(soup.find_all("gdi"), start=1):
        image_id = next((name.replace("image", "") for name in gdi.attrs if "image" in name.lower()), str(index))
        result.append({"image_id": image_id, "description": gdi.get_text().strip(),
                       "objects": [x.get_text().strip() for x in gdi.find_all("gdo")],
                       "actions": [x.get_text().strip() for x in gdi.find_all("gda")],
                       "locations": [x.get_text().strip() for x in gdi.find_all("gdl")]})
    return result


def _markdown_rows(block: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []
    headers = [value.strip() for value in lines[0].strip("|").split("|")]
    return [dict(zip(headers, [value.strip() for value in line.strip("|").split("|")]))
            for line in lines[2:] if len(line.strip("|").split("|")) == len(headers)]


def parse_cot_grounding(text: str) -> dict[int, dict[str, list[dict[str, Any]]]]:
    frames: dict[int, dict[str, list[dict[str, Any]]]] = {}
    matches = list(re.finditer(r"^##\s*Image\s+(\d+)", text or "", re.MULTILINE))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end():end]
        frame = {"characters": [], "objects": []}
        for label, key, id_name in (("Characters", "characters", "Character ID"), ("Objects", "objects", "Object ID")):
            table = re.search(rf"###\s*{label}(.*?)(?=\n###|\n##|$)", section, re.DOTALL)
            if not table:
                continue
            for row in _markdown_rows(table.group(1)):
                try:
                    bbox = [int(value) for value in row.get("Bounding Box", "").split(",")]
                except ValueError:
                    continue
                if len(bbox) == 4 and row.get(id_name):
                    frame[key].append({"id": row[id_name], "bbox": bbox})
        frames[int(match.group(1)) - 1] = frame
    return frames


