"""Dataset parsing and data loaders for StoryReasoning."""

from __future__ import annotations

import random
import re
from typing import Any

import torch
from bs4 import BeautifulSoup
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


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


def _crop_and_resize(image: Any, bbox: list[int], image_hw: tuple[int, int]) -> torch.Tensor:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, min(x1, width - 1)), max(1, min(x2, width))
    y1, y2 = max(0, min(y1, height - 1)), max(1, min(y2, height))
    return transforms.ToTensor()(transforms.Resize(image_hw)(image.crop((x1, y1, x2, y2))))


class SequencePredictionDataset(Dataset):
    def __init__(self, dataset: Any, tokenizer: Any, frames: int = 4,
                 max_length: int = 120, image_hw: tuple[int, int] = (60, 125)) -> None:
        self.dataset, self.tokenizer = dataset, tokenizer
        self.frames, self.max_length, self.image_hw = frames, max_length, image_hw
        self.transform = transforms.Compose([transforms.Resize(image_hw), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]
        attributes = parse_gdi_text(row["story"])
        cot_frames = parse_cot_grounding(row.get("chain_of_thought", ""))
        frame_tensors, descriptions = [], []
        for frame_index in range(self.frames):
            frame_tensors.append(self.transform(TF.equalize(row["images"][frame_index])))
            descriptions.append(self.tokenizer(attributes[frame_index]["description"], return_tensors="pt",
                                               padding="max_length", truncation=True,
                                               max_length=self.max_length).input_ids.squeeze(0))
        target_image = self.transform(TF.equalize(row["images"][self.frames]))
        target_ids = self.tokenizer(attributes[self.frames]["description"], return_tensors="pt",
                                    padding="max_length", truncation=True,
                                    max_length=self.max_length).input_ids
        roi1, roi2 = torch.zeros((3, *self.image_hw)), torch.zeros((3, *self.image_hw))
        roi_valid, roi_frame, entity_id = torch.tensor(0), torch.tensor(-1), ""
        detections: dict[str, list[tuple[int, list[int]]]] = {}
        for frame_index, frame in cot_frames.items():
            for detection in frame["characters"] + frame["objects"]:
                detections.setdefault(detection["id"], []).append((frame_index, detection["bbox"]))
        candidates = [(entity, items) for entity, items in detections.items() if len(items) >= 2]
        if candidates:
            entity_id, selected = random.choice(candidates)
            (frame1, bbox1), (frame2, bbox2) = random.sample(selected, 2)
            if frame1 < self.frames and frame2 < self.frames:
                roi1, roi2 = (_crop_and_resize(row["images"][frame1], bbox1, self.image_hw),
                              _crop_and_resize(row["images"][frame2], bbox2, self.image_hw))
                roi_valid, roi_frame = torch.tensor(1), torch.tensor(frame1)
        return (torch.stack(frame_tensors), torch.stack(descriptions), target_image, target_ids,
                roi1, roi2, roi_valid, roi_frame, entity_id)


class TextTaskDataset(Dataset):
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> str:
        return random.choice(parse_gdi_text(self.dataset[index]["story"])[:5])["description"]


class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset: Any, image_hw: tuple[int, int] = (60, 125)) -> None:
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.Resize(image_hw), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        return (self.transform(random.choice(self.dataset[index]["images"])),)