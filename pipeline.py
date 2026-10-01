#!/usr/bin/env python3
"""Offline-first extraction pipeline for raster P&IDs and engineering tables."""

from __future__ import annotations

import argparse
import csv
import difflib
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise SystemExit("Install runtime dependencies with: pip install Pillow numpy") from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_DPI = 360
DEFAULT_CONFIG = ROOT / "config/pipeline_config.json"


def load_config(path: str | Path | None = None) -> dict:
    """Load runtime thresholds without coupling extraction to an answer key."""
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"pipeline config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


@dataclass
class OCRItem:
    text: str
    confidence: float
    left: float
    top: float
    right: float
    bottom: float
    alternatives: list[dict]

    @property
    def cx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2


def run(command: Sequence[str], *, capture: bool = True) -> str:
    proc = subprocess.run(
        list(map(str, command)),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return proc.stdout if capture else ""


def find_binary(name: str) -> str | None:
    direct = shutil.which(name)
    if direct:
        return direct
    patterns = [
        str(Path.home() / ".cache/codex-runtimes/*/dependencies/bin" / name),
        str(Path.home() / ".cache/codex-runtimes/*/*/dependencies/bin" / name),
        str(Path.home() / ".cache/codex-runtimes/*/dependencies/native/*/bin" / name),
        str(Path.home() / ".cache/codex-runtimes/*/dependencies/bin/override" / name),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def render_input(input_path: Path, work: Path, dpi: int) -> list[Path]:
    if input_path.suffix.lower() != ".pdf":
        return [input_path]
    pdftoppm = find_binary("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm is required for PDFs (install Poppler)")
    prefix = work / "page"
    run([pdftoppm, "-png", "-r", str(dpi), input_path, prefix], capture=False)
    pages = sorted(work.glob("page-*.png"))
    if not pages:
        raise RuntimeError("PDF renderer produced no pages")
    return pages


class OCRBackend:
    def __init__(self, work: Path, requested: str = "auto") -> None:
        self.work = work
        self.kind = self._choose(requested)
        self.executable = self._prepare()

    def _choose(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if sys.platform == "darwin" and shutil.which("clang"):
            return "vision"
        if shutil.which("tesseract"):
            return "tesseract"
        raise RuntimeError("No OCR backend found. On macOS use Apple Vision; elsewhere install Tesseract.")

    def _prepare(self) -> str:
        if self.kind == "tesseract":
            binary = shutil.which("tesseract")
            if not binary:
                raise RuntimeError("Tesseract backend requested but tesseract is not installed")
            return binary
        if self.kind != "vision":
            raise RuntimeError(f"unknown OCR backend: {self.kind}")
        source = ROOT / "scripts/vision_ocr.m"
        binary = self.work / "vision_ocr"
        module_cache = self.work / "clang-cache"
        module_cache.mkdir(exist_ok=True)
        command = [
            "clang", "-O2", "-fobjc-arc", f"-fmodules-cache-path={module_cache}",
            "-framework", "Foundation", "-framework", "Vision",
            "-framework", "ImageIO", "-framework", "CoreGraphics",
            source, "-o", binary,
        ]
        run(command, capture=True)
        return str(binary)

    def recognize(self, image_path: Path) -> list[dict]:
        if self.kind == "vision":
            return json.loads(run([self.executable, image_path]))
        tsv = run([self.executable, image_path, "stdout", "--psm", "11", "tsv"])
        rows = list(csv.DictReader(tsv.splitlines(), delimiter="\t"))
        with Image.open(image_path) as image:
            width, height = image.size
        output = []
        for row in rows:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            left, top = int(row["left"]), int(row["top"])
            w, h = int(row["width"]), int(row["height"])
            output.append({
                "text": text,
                "confidence": max(0.0, float(row["conf"])) / 100,
                "x": left / width,
                "y_top": top / height,
                "width": w / width,
                "height": h / height,
                "alternatives": [],
            })
        return output


def tile_boxes(width: int, height: int, columns: int = 3, rows: int = 3, overlap: float = 0.04):
    for row in range(rows):
        for column in range(columns):
            x0 = max(0, int((column / columns - overlap) * width))
            x1 = min(width, int(((column + 1) / columns + overlap) * width))
            y0 = max(0, int((row / rows - overlap) * height))
            y1 = min(height, int(((row + 1) / rows + overlap) * height))
            yield x0, y0, x1, y1


def recognize_tiled(image_path: Path, backend: OCRBackend, work: Path, config: dict | None = None) -> list[OCRItem]:
    ocr_config = (config or {}).get("ocr", {})
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    items: list[OCRItem] = []
    boxes = tile_boxes(
        width,
        height,
        int(ocr_config.get("tile_columns", 3)),
        int(ocr_config.get("tile_rows", 3)),
        float(ocr_config.get("tile_overlap", 0.04)),
    )
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        tile_path = work / f"{image_path.stem}-tile-{index}.png"
        image.crop((x0, y0, x1, y1)).save(tile_path)
        tw, th = x1 - x0, y1 - y0
        for raw in backend.recognize(tile_path):
            left = x0 + float(raw["x"]) * tw
            if "y_top" in raw:
                top = y0 + float(raw["y_top"]) * th
            else:  # Vision coordinates have their origin at bottom-left.
                top = y0 + (1 - float(raw["y"]) - float(raw["height"])) * th
            right = left + float(raw["width"]) * tw
            bottom = top + float(raw["height"]) * th
            items.append(OCRItem(
                text=raw["text"].strip(), confidence=float(raw.get("confidence", 0)),
                left=left / width, top=top / height, right=right / width,
                bottom=bottom / height, alternatives=raw.get("alternatives", []),
            ))
    return deduplicate_ocr(items, float(ocr_config.get("dedup_distance", 0.018)))


def deduplicate_ocr(items: list[OCRItem], distance_threshold: float = 0.018) -> list[OCRItem]:
    kept: list[OCRItem] = []
    def priority(item: OCRItem) -> tuple[int, float]:
        specialized = any(alt.get("source") in ("instrument_bubble", "continuation_hexagon") for alt in item.alternatives)
        return int(specialized), item.confidence

    for item in sorted(items, key=priority, reverse=True):
        key = re.sub(r"\W", "", item.text).upper()
        duplicate = False
        for other in kept:
            other_key = re.sub(r"\W", "", other.text).upper()
            distance = math.hypot(item.cx - other.cx, item.cy - other.cy)
            if key and key == other_key and distance < distance_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return sorted(kept, key=lambda i: (i.top, i.left))


def uniq(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalized_text(text: str) -> str:
    table = str.maketrans({"Ø": "0", "О": "0", "І": "I", "|": "I", "Т": "T", "–": "-", "—": "-", "_": "-"})
    return re.sub(r"\s+", " ", text.upper().translate(table)).strip()


def all_candidate_text(item: OCRItem) -> list[str]:
    return uniq([item.text] + [alt.get("text", "") for alt in item.alternatives])


def profile_settings(config: dict | None, profile_name: str | None) -> dict:
    profiles = (config or {}).get("document_profiles", {}).get("profiles", {})
    selected = profile_name or (config or {}).get("document_profiles", {}).get("default", "legacy_nuclear_pid")
    return profiles.get(selected, {})


def select_document_profile(items: list[OCRItem], config: dict | None, requested: str = "auto") -> str:
    """Choose a tag grammar from OCR evidence without consulting an answer key."""
    profile_config = (config or {}).get("document_profiles", {})
    profiles = profile_config.get("profiles", {})
    if requested != "auto":
        if requested not in profiles:
            raise ValueError(f"unknown document profile: {requested}")
        return requested
    auto = profile_config.get("auto", {})
    def matched_signals(pattern_key: str) -> set[str]:
        patterns = [re.compile(pattern, re.I) for pattern in auto.get(pattern_key, [])]
        signals = set()
        for item in items:
            for candidate in all_candidate_text(item):
                text = normalized_text(candidate)
                for pattern in patterns:
                    signals.update(match.group(0).replace(" ", "") for match in pattern.finditer(text))
        return signals

    generic_signals = matched_signals("generic_signal_patterns")
    if len(generic_signals) >= int(auto.get("generic_min_unique_signals", 3)):
        return "generic_isa_pid"
    legacy_signals = matched_signals("legacy_signal_patterns")
    if len(legacy_signals) >= int(auto.get("legacy_min_unique_signals", 1)):
        return "legacy_nuclear_pid"
    return auto.get("unknown_profile", "unknown_pid")


def has_source(item: OCRItem, source: str) -> bool:
    return any(alt.get("source") == source for alt in item.alternatives)


def tags(
    items: list[OCRItem],
    prefix: str,
    minimum_digits: int = 2,
    maximum_digits: int = 4,
    require_source: str | None = None,
    allow_wrapped_anchor: bool = False,
) -> list[str]:
    found = []
    pattern = re.compile(rf"\b{re.escape(prefix)}[\s-]*(\d{{{minimum_digits},{maximum_digits}}})\b", re.I)
    for item in items:
        if require_source and not has_source(item, require_source):
            continue
        for candidate in all_candidate_text(item):
            candidate_text = normalized_text(candidate)
            if prefix.upper() == "PCV":
                candidate_text = re.sub(r"\bPC[YV]\b", "PCV", candidate_text)
            for number in pattern.findall(candidate_text):
                found.append(prefix.upper() + number)
    # Instrument bubbles are often read as two lines (type above, number below).
    for anchor in items:
        if require_source and not has_source(anchor, require_source):
            continue
        candidates = [normalized_text(value).replace("PCY", "PCV") for value in all_candidate_text(anchor)]
        if not any(
            re.fullmatch(re.escape(prefix), value)
            or (allow_wrapped_anchor and re.sub(r"[^A-Z0-9]", "", value) == prefix.upper())
            for value in candidates
        ):
            continue
        for candidate in nearby(items, anchor, .025):
            for value in all_candidate_text(candidate):
                split_minimum = 3 if minimum_digits <= 2 else minimum_digits
                match = re.fullmatch(rf"\D*(\d{{{split_minimum},{maximum_digits}}})\D*", normalized_text(value))
                if match:
                    found.append(prefix.upper() + match.group(1))
                    break
    return uniq(found)


def nearby(items: list[OCRItem], anchor: OCRItem, radius: float = 0.09) -> list[OCRItem]:
    return sorted(
        [i for i in items if i is not anchor and math.hypot(i.cx - anchor.cx, i.cy - anchor.cy) <= radius],
        key=lambda i: math.hypot(i.cx - anchor.cx, i.cy - anchor.cy),
    )


def items_in_region(items: list[OCRItem], region: Sequence[float]) -> list[OCRItem]:
    """Keep OCR evidence whose center lies inside a normalized page region."""
    left, top, right, bottom = map(float, region)
    return [item for item in items if left <= item.cx <= right and top <= item.cy <= bottom]


def direct_tags(
    items: list[OCRItem], prefix: str, minimum_digits: int, maximum_digits: int
) -> list[str]:
    """Read complete tags from one OCR box, excluding spatially joined fragments."""
    pattern = re.compile(rf"\b{re.escape(prefix)}[\s-]*(\d{{{minimum_digits},{maximum_digits}}})\b", re.I)
    return uniq(
        prefix.upper() + number
        for item in items
        for candidate in all_candidate_text(item)
        for number in pattern.findall(normalized_text(candidate))
    )


def prefer_direct_near_matches(candidates: list[str], direct: list[str]) -> list[str]:
    """Suppress a split-OCR variant when a complete tag differs by one digit."""
    output = []
    for value in candidates:
        replacement = next(
            (
                trusted for trusted in direct
                if len(trusted) == len(value)
                and sum(a != b for a, b in zip(trusted, value)) <= 1
            ),
            value,
        )
        if replacement not in output:
            output.append(replacement)
    return output


def context_linked_tags(
    items: list[OCRItem],
    output_prefix: str,
    context_prefixes: Sequence[str],
    minimum_digits: int,
    maximum_digits: int,
    radius: float,
) -> list[str]:
    """Recover a lost bubble prefix from an adjacent, matching context tag."""
    if not context_prefixes:
        return []
    context_pattern = re.compile(
        rf"\b(?:{'|'.join(map(re.escape, context_prefixes))})[- ]?(\d{{{minimum_digits},{maximum_digits}}})\b"
    )
    plain_pattern = re.compile(rf"\D*(\d{{{minimum_digits},{maximum_digits}}})\D*")
    output = []
    for anchor in items:
        context_numbers = [
            number
            for value in all_candidate_text(anchor)
            for number in context_pattern.findall(normalized_text(value))
        ]
        for number in context_numbers:
            if any(
                (match := plain_pattern.fullmatch(normalized_text(value))) is not None
                and match.group(1) == number
                for candidate in nearby(items, anchor, radius)
                for value in all_candidate_text(candidate)
            ):
                output.append(output_prefix.upper() + number)
    return uniq(output)


def extract_pid(
    items: list[OCRItem],
    config: dict | None = None,
    profile_name: str = "legacy_nuclear_pid",
) -> dict[str, list[str] | str]:
    profile = profile_settings(config, profile_name)
    texts = [normalized_text(i.text) for i in items]
    joined = " \n ".join(texts)
    if profile_name == "unknown_pid":
        drawing_patterns = [
            re.compile(pattern, re.I)
            for pattern in profile.get("drawing_number_patterns", [r"\b[A-Z]{1,3}-\d{4,6}\b"])
        ]
        drawing_candidates = []
        for anchor in items:
            if "DRAWING NUMBER" not in normalized_text(anchor.text):
                continue
            for candidate in nearby(items, anchor, .075):
                for value in all_candidate_text(candidate):
                    normalized = normalized_text(value)
                    for pattern in drawing_patterns:
                        for match in pattern.finditer(normalized):
                            drawing_candidates.append((math.hypot(candidate.cx - anchor.cx, candidate.cy - anchor.cy), match.group(0)))
        if drawing_candidates:
            return {"pid": min(drawing_candidates)[1]}
        return {}
    if profile_name == "generic_isa_pid":
        direct_ids = re.findall(r"\b[A-Z]{2}\d{3,6}-P-\d{4,6}\b", joined)
        split_ids = []
        drawing_left_pattern = profile.get("drawing_left_pattern", r"[A-Z]{2}\d{3,6}-?")
        drawing_right_pattern = profile.get("drawing_right_pattern", r"P-\d{4,6}")
        left_pattern = re.compile(rf"\b{drawing_left_pattern}\b")
        right_pattern = re.compile(rf"\b{drawing_right_pattern}\b")
        for anchor in items:
            left_matches = []
            for value in all_candidate_text(anchor):
                left_matches.extend(left_pattern.findall(normalized_text(value)))
            for left in left_matches:
                for candidate in nearby(items, anchor, .075):
                    for value in all_candidate_text(candidate):
                        right = right_pattern.search(normalized_text(value))
                        if right:
                            split_ids.append(f"{left.rstrip('-')}-{right.group(0)}")
        pid = uniq(direct_ids + split_ids)[-1] if direct_ids or split_ids else ""
        valve_prefixes = profile.get("valve_prefixes", ["BV", "DV", "CHV", "PRV", "SV", "BNV"])
        valve_pattern = re.compile(
            rf"\b(?:{'|'.join(map(re.escape, sorted(valve_prefixes, key=len, reverse=True)))})[- ]?\d{{5,6}}[A-Z]?\b"
        )
        valves = uniq(
            match.group(0).replace(" ", "")
            for item in items
            for value in all_candidate_text(item)
            for match in valve_pattern.finditer(normalized_text(value))
        )
    else:
        pid_matches = re.findall(r"P[I1L]D[- ]*1[- ]*([A-Z]{2})[- ]*LR\s*(\d{4,6})", joined)
        pid = f"PID-1-{pid_matches[-1][0]}_LR{pid_matches[-1][1]}" if pid_matches else ""
        valves = tags(items, "V")
    bubble_pcvs = [item for item in items if "PCV" in normalized_text(item.text) and any(alt.get("source") == "instrument_bubble" for alt in item.alternatives)]
    long_series: set[int] = set()
    for anchor in bubble_pcvs:
        group = sorted([candidate for candidate in bubble_pcvs if abs(candidate.cx - anchor.cx) < .012], key=lambda candidate: candidate.cy)
        distinct = []
        for candidate in group:
            if not distinct or candidate.cy - distinct[-1].cy > .04:
                distinct.append(candidate)
        if len(distinct) >= 4:
            gaps = [right.cy - left.cy for left, right in zip(distinct, distinct[1:])]
            regulator_context = [candidate for candidate in group if any("VAR" in normalized_text(context.text) or "REG. ELEMENT" in normalized_text(context.text) for context in nearby(items, candidate, .055))]
            if sum(.12 <= gap <= .22 for gap in gaps) >= 3 and len(regulator_context) >= 3:
                long_series.update(id(candidate) for candidate in regulator_context)
    u_bend_items = []
    for item in bubble_pcvs:
        aligned = [candidate for candidate in bubble_pcvs if candidate is item or (abs(candidate.cx - item.cx) < .012 and .05 <= abs(candidate.cy - item.cy) <= .16)]
        has_seed = any(any(alt.get("u_bend") for alt in candidate.alternatives) for candidate in aligned)
        note_context = any(any("NOTE 8" in normalized_text(context.text) for context in nearby(items, candidate, .06)) for candidate in aligned)
        if (len(aligned) >= 2 and has_seed and note_context) or id(item) in long_series:
            u_bend_items.append(item)
    pcvs = tags(u_bend_items, "PCV")
    instrument_minimum, instrument_maximum = map(int, profile.get("instrument_digits", [3, 4]))
    required_source = "instrument_bubble" if profile.get("instrument_tags_require_bubble") else None
    if profile_name == "legacy_nuclear_pid":
        # Preserve the original broad OCR path byte-for-byte for the benchmark
        # family; the generic profile uses stricter symbol-gated tags below.
        pis = tags(items, "PI")
        tis = tags(items, "TI")
    else:
        process_items = items_in_region(items, profile.get("bubble_region", [0.0, 0.0, 1.0, 1.0]))
        # Generic drawings often split a bubble into two OCR boxes ("PI" and
        # "510544"). The process-region guard avoids title-block/control-panel
        # numbers while preserving complete symbol OCR when available.
        pis = uniq(
            tags(items, "PI", instrument_minimum, instrument_maximum, required_source)
            + tags(process_items, "PI", instrument_minimum, instrument_maximum, allow_wrapped_anchor=True)
            + context_linked_tags(
                process_items,
                "PI",
                profile.get("pressure_indicator_context_prefixes", []),
                instrument_minimum,
                instrument_maximum,
                float(profile.get("pressure_indicator_context_radius", 0.04)),
            )
        )
        tis = uniq(
            tags(items, "TI", instrument_minimum, instrument_maximum, required_source)
            + tags(process_items, "TI", instrument_minimum, instrument_maximum, allow_wrapped_anchor=True)
        )

    compound = []
    for anchor in items:
        if "COMPOUND" in normalized_text(anchor.text):
            candidates = []
            for candidate in [anchor] + nearby(items, anchor, .08):
                distance = math.hypot(candidate.cx - anchor.cx, candidate.cy - anchor.cy)
                direct = re.findall(r"\bPI\s*(\d{4})\b", normalized_text(candidate.text))
                plain = re.findall(r"\b(\d{4})\b", normalized_text(candidate.text))
                for number in direct or plain:
                    dx, dy = candidate.cx - anchor.cx, candidate.cy - anchor.cy
                    # A descriptive label is normally placed immediately above
                    # or beside the instrument it qualifies. Prefer a PI bubble
                    # below and horizontally aligned with COMPOUND GAUGE; pure
                    # nearest-neighbour distance can select an adjacent gauge.
                    aligned_below = .004 <= dy <= .06 and abs(dx) <= .018
                    directional_score = abs(dx) + .25 * dy if aligned_below else distance + .1
                    candidates.append((not aligned_below, directional_score - (.02 if direct else 0), number))
            if candidates:
                compound.append("PI" + min(candidates)[2])

    pumps = []
    if profile_name == "generic_isa_pid":
        configured_pump_pattern = profile.get("pump_pattern", r"P\d{5,6}[A-Z]?")
        pump_pattern = re.compile(rf"\b{configured_pump_pattern}\b")
        region = profile.get("bubble_region", [0.0, 0.0, 1.0, 1.0])
        pump_context_pattern = re.compile(profile.get("pump_context_pattern", r"\b(?:METERING|DOSING)\s+PUMP\b"))
        pump_contexts = [
            item for item in items
            if any(pump_context_pattern.search(normalized_text(value)) for value in all_candidate_text(item))
        ]
        context_radius = float(profile.get("pump_context_radius", 0.09))
        for item in items:
            if not (region[0] <= item.cx <= region[2] and region[1] <= item.cy <= region[3]):
                continue
            for value in all_candidate_text(item):
                for match in pump_pattern.finditer(normalized_text(value)):
                    if any(math.hypot(item.cx - anchor.cx, item.cy - anchor.cy) <= context_radius for anchor in pump_contexts):
                        pumps.append(match.group(0))
        pumps = uniq(pumps)
    else:
        for anchor in items:
            if "VAC" in normalized_text(anchor.text) or "PUMP" in normalized_text(anchor.text):
                for candidate in [anchor] + nearby(items, anchor, .07):
                    pumps.extend(re.findall(r"\bP[- ](\d{2,4})\b", normalized_text(candidate.text)))
        pumps = ["P-" + value for value in uniq(pumps)]

    filters = []
    for item in items:
        filters.extend("F-" + n for n in re.findall(r"\bF[- ]?(\d{2,4})\b", normalized_text(item.text)))

    hoses = []
    for anchor in items:
        if "FLEX HOSE" not in normalized_text(anchor.text):
            continue
        group = [anchor] + nearby(items, anchor, .055)
        block = " ".join(normalized_text(value) for i in sorted(group, key=lambda i: (i.top, i.left)) for value in all_candidate_text(i))
        material = re.search(r"(\d[- ]SS)", block)
        number = re.search(r"NO\.?\s*([1-9])", block)
        if material and number:
            hoses.append(f"{material.group(1).replace(' ', '-')}-FLEX HOSE NO. {number.group(1)}")

    tubing = []
    for text in texts:
        for numerator, denominator in re.findall(r"([13])\s*\/\s*([248])\s*[-°'\"]{0,2}\s*T\b", text):
            tubing.append(f"{numerator}/{denominator}-T")

    gases = []
    for gas in ("NITROGEN", "ARGON", "HELIUM", "HYDROGEN", "OXYGEN"):
        if any(gas in text for text in texts):
            gases.append(gas)

    def named_equipment(label: str, anchor_phrase: str, id_pattern: str) -> list[str]:
        output = []
        for anchor in items:
            if anchor_phrase not in normalized_text(anchor.text):
                continue
            group = [anchor] + nearby(items, anchor, .08)
            identifier = None
            for candidate in group:
                for value in all_candidate_text(candidate):
                    match = re.search(id_pattern, normalized_text(value))
                    if match:
                        identifier = match.group(0).replace(" ", "-")
                        break
                if identifier:
                    break
            if identifier:
                output.append(f"{label} {identifier}")
        return uniq(output)

    panels = named_equipment("PRIMARY SAMPLE PANEL", "SAMPLE PANEL", r"CP[- ]\d+[A-Z]?")
    sinks = named_equipment("PRIMARY SAMPLE SINK", "SAMPLE SINK", r"MM[- ]\d+[A-Z]?")

    boundaries = []
    boundary_map = {"SAMPLE SYSTEM": "SS", "DEMINERALIZED WATER SYSTEM": "DM", "CHEMICAL & VOLUME CONTROL SYSTEM": "CS"}
    for phrase, code in boundary_map.items():
        if any(phrase in text for text in texts):
            boundaries.append(code)

    demin_candidates = []
    for item in items:
        item_text = normalized_text(item.text)
        if "DEMINERALIZED WATER" in item_text or "MINERIALIZE" in item_text or "MINERALIZE" in item_text:
            for candidate in [item] + nearby(items, item, .06):
                distance = math.hypot(candidate.cx - item.cx, candidate.cy - item.cy)
                for number in re.findall(r"\b(?:Z\s*)?(\d{5,6})\b", normalized_text(candidate.text)):
                    demin_candidates.append((number, distance))
    demin = []
    if demin_candidates:
        counts = {number: sum(1 for value, _ in demin_candidates if value == number) for number, _ in demin_candidates}
        demin = [min(counts, key=lambda number: (-counts[number], min(distance for value, distance in demin_candidates if value == number)))]

    continuations = []
    for item in items:
        if not any(alt.get("source") == "continuation_hexagon" for alt in item.alternatives):
            continue
        text = re.sub(r"[^A-Z0-9]", "", normalized_text(item.text))
        if re.fullmatch(r"(?:[2-9]|1[0-9])[A-Z]?", text):
            continuations.append(text)

    generic_answers: dict[str, list[str]] = {}
    if profile_name == "generic_isa_pid":
        process_items = items_in_region(items, profile.get("bubble_region", [0.0, 0.0, 1.0, 1.0]))
        flow_candidates = tags(process_items, "FI", instrument_minimum, instrument_maximum)
        flow_direct = direct_tags(process_items, "FI", instrument_minimum, instrument_maximum)
        generic_answers = {
            "flow_indicators": prefer_direct_near_matches(flow_candidates, flow_direct),
            "flow_elements": tags(process_items, "FE", instrument_minimum, instrument_maximum),
            "flow_indicator_transmitters": tags(process_items, "FIT", instrument_minimum, instrument_maximum),
            "flow_switches": tags(process_items, "FS", instrument_minimum, instrument_maximum),
            "pressure_relief_valves": [value for value in valves if value.startswith("PRV")],
            "solenoid_valves": [value for value in valves if value.startswith("SV")],
            "check_valves": [value for value in valves if value.startswith("CHV")],
        }

    answers: dict[str, list[str] | str] = {
        "pid": pid,
        "valves": valves,
        "pressure_control_valves (U-bend_pipe)": pcvs,
        "compound_gauge": uniq(compound),
        "pressure_indicators": pis,
        "temperature_indicators": tis,
        "vacuum_pumps": pumps,
        "filters": uniq(filters),
        "flexible_hoses": uniq(hoses),
        "tubing_sizes": uniq(tubing),
        "gas_cylinders": gases,
        "sample_panels": panels,
        "sample_sinks": sinks,
        "continuation_connections (hexagon)": uniq(continuations),
        "system_boundary_interfaces": boundaries,
        "instrument_bubbles": [name for name, vals in (("PI", pis), ("TI", tis), ("PCV", pcvs)) if vals],
        "demineralized_water": ["Z" + n for n in uniq(demin)],
    }
    answers.update(generic_answers)
    return {key: value for key, value in answers.items() if value not in ([], "")}


def _format_match(category: str, value: str) -> float:
    patterns = {
        "pid": r"PID-1-[A-Z]{2}_LR\d{4,6}",
        "valves": r"(?:V\d{2,5}|(?:BV|DV|CHV|PRV|SV|BNV)\d{5,6}[A-Z]?)",
        "pressure_control_valves (U-bend_pipe)": r"PCV\d{3,6}",
        "compound_gauge": r"PI\d{3,6}",
        "pressure_indicators": r"PI\d{3,6}",
        "temperature_indicators": r"TI\d{3,6}",
        "vacuum_pumps": r"(?:P-\d{2,4}|P\d{5,6}[A-Z]?)",
        "filters": r"F-\d{1,4}",
        "flow_indicators": r"FI\d{5,6}",
        "flow_elements": r"FE\d{5,6}",
        "flow_indicator_transmitters": r"FIT\d{5,6}",
        "flow_switches": r"FS\d{5,6}",
        "pressure_relief_valves": r"PRV\d{5,6}[A-Z]?",
        "solenoid_valves": r"SV\d{5,6}[A-Z]?",
        "check_valves": r"CHV\d{5,6}[A-Z]?",
        "continuation_connections (hexagon)": r"(?:[2-9]|1[0-9])[A-Z]?",
        "demineralized_water": r"Z\d{5,6}",
    }
    pattern = patterns.get(category)
    return 1.0 if pattern and re.fullmatch(pattern, value) else (0.7 if not pattern else 0.0)


def _matching_evidence(value: str, category: str, items: list[OCRItem]) -> list[OCRItem]:
    target = re.sub(r"[^A-Z0-9]", "", normalized_text(value))
    matches = []
    for item in items:
        specialized_sources = {alt.get("source") for alt in item.alternatives}
        text = re.sub(r"[^A-Z0-9]", "", normalized_text(item.text))
        candidate_texts = [text] + [re.sub(r"[^A-Z0-9]", "", normalized_text(alt.get("text", ""))) for alt in item.alternatives]
        if category == "continuation_connections (hexagon)":
            if "continuation_hexagon" in specialized_sources and text == target:
                matches.append(item)
            continue
        if target and any(
            target == candidate
            or (len(target) >= 4 and target in candidate)
            for candidate in candidate_texts if candidate
        ):
            matches.append(item)
            continue
        # Category outputs such as SS and NITROGEN can be inferred from a longer
        # legend phrase; keep those anchors available for audit.
        if category == "system_boundary_interfaces":
            phrases = {"SS": "SAMPLESYSTEM", "DM": "DEMINERALIZEDWATERSYSTEM", "CS": "CHEMICALVOLUMECONTROLSYSTEM"}
            if value in phrases and phrases[value] in text.replace("AND", ""):
                matches.append(item)
    if not matches and category in ("pressure_indicators", "temperature_indicators"):
        instrument = re.fullmatch(r"(PI|TI)(\d{3,6})", target)
        if instrument:
            prefix, number = instrument.groups()
            for anchor in items:
                anchor_texts = [re.sub(r"[^A-Z0-9]", "", normalized_text(text)) for text in all_candidate_text(anchor)]
                prefix_anchor = prefix in anchor_texts
                context_anchor = any(re.fullmatch(rf"(?:ME|IE){re.escape(number)}", text) for text in anchor_texts)
                if not (prefix_anchor or context_anchor):
                    continue
                for candidate in nearby(items, anchor, .04):
                    if any(
                        re.sub(r"[^0-9]", "", normalized_text(text)) == number
                        for text in all_candidate_text(candidate)
                    ):
                        matches.extend([anchor, candidate])
                        break
    return matches


def build_confidence_report(
    document_id: int,
    page_number: int,
    answers: dict[str, list[str] | str],
    items: list[OCRItem],
    config: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Calibrate final fields and create an actionable human-review queue."""
    settings = (config or {}).get("confidence", {})
    weights = settings.get("weights", {})
    default_threshold = float(settings.get("default_review_threshold", 0.72))
    category_thresholds = settings.get("category_review_thresholds", {})
    fields: dict[str, list[dict]] = {}
    review_queue = []
    for category, raw_values in answers.items():
        values = raw_values if isinstance(raw_values, list) else [raw_values]
        fields[category] = []
        for value in values:
            evidence = _matching_evidence(str(value), category, items)
            def evidence_priority(item: OCRItem) -> tuple[int, float]:
                specialized = any(alt.get("source") in ("instrument_bubble", "continuation_hexagon") for alt in item.alternatives)
                return int(specialized), item.confidence

            best_item = max(evidence, key=evidence_priority, default=None)
            source = "derived_rule"
            metadata: dict = {}
            if best_item:
                metadata = next((alt for alt in best_item.alternatives if alt.get("source") in ("instrument_bubble", "continuation_hexagon")), {})
                source = metadata.get("source", "page_ocr")
            ocr_confidence = float(best_item.confidence) if best_item else 0.0
            format_score = _format_match(category, str(value))
            if source == "instrument_bubble":
                confidence = (
                    float(weights.get("bubble_ocr", 0.35)) * ocr_confidence
                    + float(weights.get("bubble_vote", 0.35)) * float(metadata.get("vote_agreement", 0))
                    + float(weights.get("bubble_geometry", 0.3)) * float(metadata.get("geometry_score", 0))
                )
            elif source == "continuation_hexagon":
                confidence = (
                    float(weights.get("hexagon_ocr", 0.25)) * ocr_confidence
                    + float(weights.get("hexagon_vote", 0.3)) * float(metadata.get("vote_agreement", 0))
                    + float(weights.get("hexagon_geometry", 0.45)) * float(metadata.get("geometry_score", 0))
                )
            else:
                context_score = 1.0 if evidence else 0.0
                confidence = (
                    float(weights.get("generic_ocr", 0.65)) * ocr_confidence
                    + float(weights.get("format_match", 0.2)) * format_score
                    + float(weights.get("context", 0.15)) * context_score
                )
            source_caps = settings.get("source_caps", {}).get(source, {})
            cap = source_caps.get(category, source_caps.get("default"))
            if cap is not None:
                confidence = min(confidence, float(cap))
            confidence = round(max(0.0, min(1.0, confidence)), 4)
            threshold = float(category_thresholds.get(category, default_threshold))
            reasons = []
            if not evidence:
                reasons.append("no_direct_spatial_evidence")
            if confidence < threshold:
                reasons.append(f"confidence_below_{threshold:.2f}")
            if source in ("instrument_bubble", "continuation_hexagon") and float(metadata.get("vote_agreement", 0)) < 0.5:
                reasons.append("preprocessing_variants_disagree")
            if source == "continuation_hexagon" and int(metadata.get("ocr_vote_count", 0)) == 0:
                reasons.append("template_only_short_code")
            record = {
                "value": value,
                "confidence": confidence,
                "review_threshold": threshold,
                "source": source,
                "bbox": [round(best_item.left, 5), round(best_item.top, 5), round(best_item.right, 5), round(best_item.bottom, 5)] if best_item else None,
                "evidence_text": best_item.text if best_item else None,
                "reasons": reasons,
            }
            if metadata:
                record["signals"] = {
                    key: metadata[key] for key in (
                        "geometry_score", "vote_agreement", "u_bend", "template_score",
                        "template_margin", "ocr_vote_count",
                    ) if key in metadata
                }
            fields[category].append(record)
            if reasons:
                review_queue.append({
                    "document_id": document_id,
                    "page": page_number,
                    "category": category,
                    **record,
                })
    return {"document_id": document_id, "page": page_number, "fields": fields}, review_queue


def dark_mask(image: Image.Image, threshold: int = 150) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.min(rgb, axis=2) < threshold


def one_dimensional_runs(values: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    points = np.flatnonzero(values)
    if points.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(points) > max_gap + 1) + 1
    groups = np.split(points, cuts)
    return [(int(g[0]), int(g[-1])) for g in groups]


def line_segments(mask: np.ndarray, axis: str, minimum: int) -> list[tuple[int, int, int]]:
    count = mask.shape[0] if axis == "horizontal" else mask.shape[1]
    raw = []
    for position in range(count):
        values = mask[position, :] if axis == "horizontal" else mask[:, position]
        for start, end in one_dimensional_runs(values):
            if end - start + 1 >= minimum:
                raw.append((position, start, end))
    clusters: list[list[tuple[int, int, int]]] = []
    for segment in raw:
        pos, start, end = segment
        match = None
        for cluster in reversed(clusters[-80:]):
            last_pos, _, _ = cluster[-1]
            med_start = int(np.median([x[1] for x in cluster]))
            med_end = int(np.median([x[2] for x in cluster]))
            if pos - last_pos <= 2 and abs(start - med_start) <= 7 and abs(end - med_end) <= 7:
                match = cluster
                break
        if match is None:
            clusters.append([segment])
        else:
            match.append(segment)
    output = []
    for cluster in clusters:
        positions = [x[0] for x in cluster]
        if max(positions) - min(positions) > 14:
            continue
        output.append((int(np.median(positions)), int(np.median([x[1] for x in cluster])), int(np.median([x[2] for x in cluster]))))
    return output


def detect_instrument_bubbles(image: Image.Image, config: dict | None = None, dpi: int = DEFAULT_DPI) -> list[dict]:
    """Find circular instrument bubbles from their center divider and outline."""
    settings = (config or {}).get("instrument_bubbles", {})
    threshold = int((config or {}).get("ocr", {}).get("dark_threshold", 150))
    mask = dark_mask(image, threshold)
    height, width = mask.shape
    scale = dpi / 360
    divider_min = int(float(settings.get("divider_min_px_at_360dpi", 28)) * scale)
    divider_max = int(float(settings.get("divider_max_px_at_360dpi", 105)) * scale)
    # Allow small gaps where the type letters touch the center divider.
    horizontal = []
    for y in range(height):
        for x0, x1 in one_dimensional_runs(mask[y, :], max_gap=9):
            if x1 - x0 >= 24:
                horizontal.append((y, x0, x1))
    proposals = []
    for y, x0, x1 in horizontal:
        length = x1 - x0
        if not divider_min <= length <= min(divider_max, max(int(80 * scale), width // 45)):
            continue
        cx = (x0 + x1) / 2
        best = (0.0, 0.0)
        for factor in (1.15, 1.35, 1.55, 1.75):
            radius = length * factor / 2
            if cx - radius < 0 or cx + radius >= width or y - radius < 0 or y + radius >= height:
                continue
            hits = 0
            samples = 32
            for step in range(samples):
                angle = 2 * math.pi * step / samples
                px = int(round(cx + radius * math.cos(angle)))
                py = int(round(y + radius * math.sin(angle)))
                patch = mask[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
                hits += bool(patch.size and np.any(patch))
            score = hits / samples
            if score > best[0]:
                best = score, radius
        score, radius = best
        if score < float(settings.get("circle_score_min", 0.68)):
            continue
        bx0, by0 = int(cx - radius), int(y - radius)
        bx1, by1 = int(cx + radius), int(y + radius)
        density = float(np.mean(mask[by0:by1 + 1, bx0:bx1 + 1]))
        if not float(settings.get("density_min", 0.025)) <= density <= float(settings.get("density_max", 0.28)):
            continue
        proposals.append((score, cx, float(y), radius))
    selected = []
    for score, cx, cy, radius in sorted(proposals, key=lambda value: value[0] * value[3], reverse=True):
        if any(math.hypot(cx - ox, cy - oy) < max(radius, other_radius) * .9 for _, ox, oy, other_radius in selected):
            continue
        selected.append((score, cx, cy, radius))
    boxes = []
    for score, cx, cy, radius in selected:
        # The divider may be split by glyphs, which underestimates the radius;
        # a generous crop still stays local while retaining both text halves.
        pad = radius * float(settings.get("crop_padding_radius", 2.0))
        boxes.append({
            "bbox_px": (max(0, int(cx - pad)), max(0, int(cy - pad)), min(width, int(cx + pad)), min(height, int(cy + pad))),
            "geometry_score": round(float(score), 4),
        })
    return boxes


def classify_bubble_type(image: Image.Image, box: tuple[int, int, int, int]) -> str | None:
    crop = image.crop(box)
    mask = dark_mask(crop)
    height, width = mask.shape
    region = mask[int(.24 * height):int(.38 * height), int(.25 * width):int(.75 * width)]
    if not region.size:
        return None
    occupied = np.sum(region, axis=0) >= 2
    runs = one_dimensional_runs(occupied, max_gap=max(2, region.shape[1] // 22))
    inner_left, inner_right = .2 * region.shape[1], .8 * region.shape[1]
    central = [run for run in runs if run[1] > inner_left and run[0] < inner_right]
    if not central:
        return None
    span_ratio = (max(run[1] for run in central) - min(run[0] for run in central) + 1) / region.shape[1]
    if span_ratio >= .52:
        return "PCV"
    if len(central) < 2:
        return None
    first_x0, first_x1 = central[0]
    ys, xs = np.nonzero(region[:, first_x0:first_x1 + 1])
    lower = xs[ys >= np.median(ys)] if ys.size else np.array([])
    if lower.size == 0:
        return None
    relative = float(np.median(lower)) / max(1, first_x1 - first_x0)
    return "PI" if relative < .38 else "TI"


def has_u_bend(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    mask = dark_mask(image)
    height, width = mask.shape
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    radius = max(x1 - x0, y1 - y0) // 4
    left, right = max(0, cx - radius), min(width, cx + 4 * radius)
    top, bottom = max(0, cy + radius // 2), min(height, cy + 5 * radius)
    verticals = []
    for x in range(left, right):
        for start, end in one_dimensional_runs(mask[top:bottom, x], max_gap=2):
            length = end - start + 1
            if max(18, radius // 2) <= length <= max(180, radius * 4):
                verticals.append((x, top + start, top + end))
    clustered = []
    for segment in verticals:
        if clustered and segment[0] - clustered[-1][-1][0] <= 2 and abs(segment[1] - int(np.median([v[1] for v in clustered[-1]]))) <= 10:
            clustered[-1].append(segment)
        else:
            clustered.append([segment])
    lines = [(int(np.median([v[0] for v in group])), int(np.median([v[1] for v in group])), int(np.median([v[2] for v in group]))) for group in clustered]
    for index, (ax, ay0, ay1) in enumerate(lines):
        for bx, by0, by1 in lines[index + 1:]:
            if ax < cx + radius // 2 or min(ay0, by0) < cy + radius:
                continue
            separation = bx - ax
            if not max(10, radius // 3) <= separation <= max(80, radius * 2):
                continue
            if abs(ay0 - by0) > max(12, radius // 2) or abs(ay1 - by1) > max(16, radius // 2):
                continue
            arc_top = min(ay1, by1) - 5
            arc_bottom = min(height, max(ay1, by1) + max(14, radius))
            coverage = np.mean(np.any(mask[max(0, arc_top):arc_bottom, ax:bx + 1], axis=0))
            leader_hits = 0
            leader_samples = 18
            for step in range(leader_samples):
                fraction = step / max(1, leader_samples - 1)
                px = int(round(cx + fraction * (ax - cx)))
                py = int(round(cy + radius + fraction * (ay0 - cy - radius)))
                patch = mask[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
                leader_hits += bool(patch.size and np.any(patch))
            if coverage >= .42 and leader_hits / leader_samples >= .45:
                return True
    return False


def has_diagonal_leader(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    mask = dark_mask(image.crop(box))
    height, width = mask.shape
    region = mask[int(.68 * height):height, :]
    points = []
    for local_y in range(region.shape[0]):
        xs = np.flatnonzero(region[local_y])
        if xs.size:
            points.append((local_y, float(np.median(xs))))
    if len(points) < max(8, region.shape[0] // 5):
        return False
    ys = np.array([point[0] for point in points], dtype=float)
    xs = np.array([point[1] for point in points], dtype=float)
    slope = float(np.polyfit(ys, xs, 1)[0]) if np.ptp(ys) else 0.0
    spread = float(np.percentile(xs, 90) - np.percentile(xs, 10)) / max(1, width)
    return slope > .28 and spread > .13


def preprocess_ocr_crop(crop: Image.Image, variant: str, threshold_level: int = 185) -> Image.Image:
    """Create complementary OCR views without changing document semantics."""
    rgb = crop.convert("RGB")
    if variant == "original":
        return rgb
    gray = ImageOps.grayscale(rgb)
    if variant == "autocontrast":
        return ImageOps.autocontrast(gray, cutoff=1).filter(ImageFilter.SHARPEN).convert("RGB")
    if variant == "threshold":
        contrasted = ImageEnhance.Contrast(gray).enhance(1.6)
        return contrasted.point(lambda value: 255 if value >= threshold_level else 0).convert("RGB")
    raise ValueError(f"unknown OCR preprocessing variant: {variant}")


def majority_vote(values: list[tuple[str, float]]) -> tuple[str | None, float, float]:
    """Return winner, vote agreement, and mean OCR confidence for the winner."""
    if not values:
        return None, 0.0, 0.0
    groups: dict[str, list[float]] = {}
    for value, confidence in values:
        groups.setdefault(value, []).append(float(confidence))
    winner = max(groups, key=lambda value: (len(groups[value]), sum(groups[value]), value))
    agreement = len(groups[winner]) / len(values)
    return winner, agreement, sum(groups[winner]) / len(groups[winner])


def recognize_instrument_bubbles(
    image: Image.Image,
    backend: OCRBackend,
    work: Path,
    page_number: int,
    base_items: list[OCRItem],
    config: dict | None = None,
    dpi: int = DEFAULT_DPI,
    profile_name: str = "legacy_nuclear_pid",
) -> list[OCRItem]:
    settings = (config or {}).get("instrument_bubbles", {})
    profile = profile_settings(config, profile_name)
    detections = detect_instrument_bubbles(image, config, dpi)
    width, height = image.size
    region = profile.get("bubble_region", [0.0, 0.0, 1.0, 1.0])
    detections = [
        detection for detection in detections
        if region[0] <= (detection["bbox_px"][0] + detection["bbox_px"][2]) / (2 * width) <= region[2]
        and region[1] <= (detection["bbox_px"][1] + detection["bbox_px"][3]) / (2 * height) <= region[3]
    ]
    number_minimum, number_maximum = map(int, profile.get("bubble_number_digits", [4, 4]))
    number_pattern = re.compile(rf"\b\d{{{number_minimum},{number_maximum}}}\b")
    output = []
    batch_size = int(settings.get("batch_size", 40))
    variants = list(settings.get("preprocess_variants", ["original", "autocontrast", "threshold"]))
    threshold_level = int(settings.get("threshold_level", 185))
    effective_batch_size = max(1, batch_size // max(1, len(variants)))
    for batch_start in range(0, len(detections), effective_batch_size):
        batch = detections[batch_start:batch_start + effective_batch_size]
        slot_width, slot_height, columns = 260, 220, 5
        slot_count = len(batch) * len(variants)
        rows = math.ceil(slot_count / columns)
        sheet = Image.new("RGB", (columns * slot_width, rows * slot_height), "white")
        slots = []
        for detection_index, detection in enumerate(batch):
            box = tuple(detection["bbox_px"])
            raw_crop = image.crop(box).convert("RGB")
            for variant_index, variant in enumerate(variants):
                slot_index = detection_index * len(variants) + variant_index
                prepared = preprocess_ocr_crop(raw_crop, variant, threshold_level)
                scale = min(4.0, 198 / max(prepared.width, prepared.height))
                content = prepared.resize((max(1, int(prepared.width * scale)), max(1, int(prepared.height * scale))))
                sx = (slot_index % columns) * slot_width + (slot_width - content.width) // 2
                sy = (slot_index // columns) * slot_height + (slot_height - content.height) // 2
                sheet.paste(content, (sx, sy))
                slots.append((sx, sy, sx + content.width, sy + content.height, detection_index, variant))
        recognized = recognize_crop(sheet, backend, work / f"bubbles-{page_number}-{batch_start // effective_batch_size}.png")
        observations: dict[int, list[dict]] = {index: [] for index in range(len(batch))}
        for sx0, sy0, sx1, sy1, detection_index, variant in slots:
            pieces = [item for item in recognized if sx0 <= item.cx <= sx1 and sy0 <= item.cy <= sy1]
            pieces.sort(key=lambda item: (item.top, item.left))
            observations[detection_index].append({
                "variant": variant,
                "text": " ".join(item.text for item in pieces),
                "confidence": sum(item.confidence for item in pieces) / len(pieces) if pieces else 0.0,
            })

        for detection_index, detection in enumerate(batch):
            box = tuple(detection["bbox_px"])
            x0, y0, x1, y1 = box
            prefix_hint = classify_bubble_type(image, box)
            diagonal_leader = has_diagonal_leader(image, box)
            if prefix_hint == "PCV" and not diagonal_leader:
                prefix_hint = "PI"
            nearby_base = [item for item in base_items if x0 / width <= item.cx <= x1 / width and y0 / height <= item.cy <= y1 / height]
            number_votes: list[tuple[str, float]] = []
            prefix_votes: list[tuple[str, float]] = []
            for observation in observations[detection_index]:
                text = normalized_text(observation["text"])
                confidence = observation["confidence"]
                number_votes.extend((number, confidence) for number in number_pattern.findall(text))
                prefix_votes.extend((prefix, confidence) for prefix in re.findall(r"\b(?:PCV|PI|TI)\b", text))
            # Tiled page OCR supplies context but counts as only one vote, so a
            # single mistaken whole-page read cannot overpower crop consensus.
            for item in nearby_base:
                value = normalized_text(item.text)
                number_votes.extend((number, item.confidence) for number in number_pattern.findall(value))
                prefix_votes.extend((prefix, item.confidence) for prefix in re.findall(r"\b(?:PCV|PI|TI)\b", value))
            number, number_agreement, number_confidence = majority_vote(number_votes)
            voted_prefix, prefix_agreement, _ = majority_vote(prefix_votes)
            prefix = prefix_hint or voted_prefix
            if profile_name == "generic_isa_pid" and number and not prefix:
                # Some scans preserve the number inside a PI bubble but lose
                # the two-letter prefix. A matching measurement-element tag
                # immediately beside the same detected circle is independent
                # evidence that this is an instrument, not an arbitrary note.
                expanded_context = [
                    item for item in base_items
                    if math.hypot(item.cx - (x0 + x1) / (2 * width), item.cy - (y0 + y1) / (2 * height)) <= .045
                ]
                measurement_pattern = re.compile(rf"\b(?:ME|IE)[- ]?{re.escape(number)}\b")
                if any(
                    measurement_pattern.search(normalized_text(value))
                    for item in expanded_context
                    for value in all_candidate_text(item)
                ):
                    prefix = "PI"
                    prefix_agreement = max(prefix_agreement, 0.5)
            if prefix_hint and voted_prefix == prefix_hint:
                prefix_agreement = max(prefix_agreement, 1.0)
            u_bend = bool(prefix == "PCV" and diagonal_leader and has_u_bend(image, box))
            if not prefix or not number:
                continue
            vote_agreement = (number_agreement + prefix_agreement) / 2
            metadata = {
                "source": "instrument_bubble",
                "u_bend": u_bend,
                "geometry_score": detection["geometry_score"],
                "vote_agreement": round(vote_agreement, 4),
                "profile": profile_name,
                "ocr_variants": observations[detection_index],
            }
            output.append(OCRItem(
                f"{prefix} {number}", number_confidence,
                x0 / width, y0 / height, x1 / width, y1 / height, [metadata],
            ))
    return output


def _sample_segment(mask: np.ndarray, start: tuple[float, float], end: tuple[float, float], radius: int = 2) -> float:
    """Measure how much of an expected symbol edge is supported by ink."""
    x0, y0 = start
    x1, y1 = end
    samples = max(16, int(math.hypot(x1 - x0, y1 - y0)))
    hits = 0
    height, width = mask.shape
    for fraction in np.linspace(0, 1, samples):
        x = int(round(x0 + (x1 - x0) * fraction))
        y = int(round(y0 + (y1 - y0) * fraction))
        patch = mask[max(0, y - radius):min(height, y + radius + 1), max(0, x - radius):min(width, x + radius + 1)]
        hits += bool(patch.size and np.any(patch))
    return hits / samples


def detect_continuation_hexagons(image: Image.Image, config: dict | None = None, dpi: int = DEFAULT_DPI) -> list[dict]:
    """Detect point-left/right continuation hexagons from all six outline edges."""
    settings = (config or {}).get("continuation_hexagons", {})
    threshold = int((config or {}).get("ocr", {}).get("dark_threshold", 150))
    mask = dark_mask(image, threshold)
    height, width = mask.shape
    scale = dpi / 360
    region = settings.get("search_region", [0.08, 0.07, 0.82, 0.96])
    rx0, ry0, rx1, ry1 = region
    edge_min = int(float(settings.get("top_edge_min_px_at_360dpi", 16)) * scale)
    edge_max = int(float(settings.get("top_edge_max_px_at_360dpi", 38)) * scale)
    height_min = int(float(settings.get("height_min_px_at_360dpi", 28)) * scale)
    height_max = int(float(settings.get("height_max_px_at_360dpi", 55)) * scale)
    center_tolerance = float(settings.get("center_tolerance_px_at_360dpi", 6)) * scale
    length_ratio_min, length_ratio_max = map(float, settings.get("edge_length_ratio", [0.72, 1.38]))
    aspect_min, aspect_max = map(float, settings.get("height_to_edge_ratio", [1.05, 1.85]))
    score_min = float(settings.get("geometry_score_min", 0.9))

    horizontal = line_segments(mask, "horizontal", max(6, int(8 * scale)))
    short_edges = [
        line for line in horizontal
        if edge_min <= line[2] - line[1] + 1 <= edge_max
        and rx0 * width <= line[1] <= rx1 * width
        and ry0 * height <= line[0] <= ry1 * height
    ]
    proposals = []
    for index, (top_y, top_x0, top_x1) in enumerate(short_edges):
        top_center = (top_x0 + top_x1) / 2
        top_length = top_x1 - top_x0 + 1
        for bottom_y, bottom_x0, bottom_x1 in short_edges[index + 1:]:
            symbol_height = bottom_y - top_y
            if symbol_height > height_max:
                break
            if symbol_height < height_min:
                continue
            bottom_center = (bottom_x0 + bottom_x1) / 2
            bottom_length = bottom_x1 - bottom_x0 + 1
            if abs(top_center - bottom_center) > center_tolerance:
                continue
            if not length_ratio_min <= bottom_length / top_length <= length_ratio_max:
                continue
            average_edge = (top_length + bottom_length) / 2
            if not aspect_min <= symbol_height / average_edge <= aspect_max:
                continue
            center_x = (top_center + bottom_center) / 2
            center_y = (top_y + bottom_y) / 2
            extension = symbol_height * float(settings.get("diagonal_extension_height_ratio", 0.3))
            left_vertex = (center_x - extension - average_edge / 2, center_y)
            right_vertex = (center_x + extension + average_edge / 2, center_y)
            scores = [
                _sample_segment(mask, (top_x0, top_y), (top_x1, top_y)),
                _sample_segment(mask, (bottom_x0, bottom_y), (bottom_x1, bottom_y)),
                _sample_segment(mask, (top_x0, top_y), left_vertex),
                _sample_segment(mask, left_vertex, (bottom_x0, bottom_y)),
                _sample_segment(mask, (top_x1, top_y), right_vertex),
                _sample_segment(mask, right_vertex, (bottom_x1, bottom_y)),
            ]
            geometry_score = sum(scores) / len(scores)
            if geometry_score < score_min:
                continue
            padding = max(4, int(5 * scale))
            box = (
                max(0, int(left_vertex[0]) - padding),
                max(0, top_y - padding),
                min(width, int(right_vertex[0]) + padding),
                min(height, bottom_y + padding),
            )
            proposals.append({"bbox_px": box, "geometry_score": round(geometry_score, 4)})

    nms_distance = float(settings.get("nms_distance_px_at_360dpi", 30)) * scale
    selected = []
    for proposal in sorted(proposals, key=lambda value: value["geometry_score"], reverse=True):
        x0, y0, x1, y1 = proposal["bbox_px"]
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        if any(math.hypot(center[0] - other[0], center[1] - other[1]) < nms_distance for other in selected):
            continue
        selected.append((*center, proposal))
    return [proposal for _, _, proposal in sorted(selected, key=lambda value: (value[1], value[0]))]


def _components(mask: np.ndarray, *, keep_border: bool = False) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    output = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        points = []
        touches_border = False
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            touches_border |= y in (0, height - 1) or x in (0, width - 1)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if keep_border or not touches_border:
            output.append(points)
    return output


def _normalize_glyph(mask: np.ndarray, target_height: int = 48, target_width: int = 64) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    output = np.zeros((target_height, target_width), dtype=bool)
    if not len(xs):
        return output
    cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = min((target_width - 8) / cropped.shape[1], (target_height - 8) / cropped.shape[0])
    resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize((
        max(1, int(round(cropped.shape[1] * scale))),
        max(1, int(round(cropped.shape[0] * scale))),
    ))
    normalized = np.asarray(resized) > 80
    y0 = (target_height - normalized.shape[0]) // 2
    x0 = (target_width - normalized.shape[1]) // 2
    output[y0:y0 + normalized.shape[0], x0:x0 + normalized.shape[1]] = normalized
    return output


def _glyph_mask(crop: Image.Image, threshold: int = 180) -> np.ndarray:
    gray = np.asarray(ImageOps.grayscale(crop))
    raw = gray < threshold
    minimum_area = max(4, int(raw.size * 0.0025))
    components = [component for component in _components(raw) if len(component) >= minimum_area]
    components.sort(key=len, reverse=True)
    output = np.zeros_like(raw, dtype=bool)
    for component in components[:3]:
        for y, x in component:
            output[y, x] = True
    return _normalize_glyph(output)


def _dilate(mask: np.ndarray) -> np.ndarray:
    output = mask.copy()
    padded = np.pad(mask, 1)
    for dy in range(3):
        for dx in range(3):
            output |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return output


def _hole_count(mask: np.ndarray) -> int:
    holes = 0
    for component in _components(~mask, keep_border=True):
        if len(component) < 12:
            continue
        if any(y in (0, mask.shape[0] - 1) or x in (0, mask.shape[1] - 1) for y, x in component):
            continue
        holes += 1
    return holes


def _glyph_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if not np.any(left) or not np.any(right):
        return 0.0
    tolerant_left = np.logical_and(left, _dilate(right)).sum() / max(1, left.sum())
    tolerant_right = np.logical_and(right, _dilate(left)).sum() / max(1, right.sum())
    hole_penalty = 0.08 * abs(_hole_count(left) - _hole_count(right))
    return max(0.0, (tolerant_left + tolerant_right) / 2 - hole_penalty)


def classify_short_code(crop: Image.Image, settings: dict) -> tuple[str | None, float, float]:
    """Recognize 1-2 character continuation labels when general OCR abstains."""
    label_min, label_max = map(int, settings.get("template_label_range", [2, 19]))
    suffixes = list(settings.get("template_suffixes", ["", "A"]))
    labels = [f"{number}{suffix}" for number in range(label_min, label_max + 1) for suffix in suffixes]
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "DejaVuSans.ttf",
    ]
    fonts = []
    for path in font_candidates:
        try:
            fonts.append(ImageFont.truetype(path, 44))
        except OSError:
            continue
    if not fonts:
        fonts = [ImageFont.load_default(size=40)]
    observed_masks = [_glyph_mask(crop, threshold) for threshold in (155, 180, 205)]
    scores = []
    for label in labels:
        best = 0.0
        for font in fonts:
            template_image = Image.new("L", (120, 70), 255)
            ImageDraw.Draw(template_image).text((4, 0), label, font=font, fill=0)
            template = _normalize_glyph(np.asarray(template_image) < 128)
            best = max(best, *(_glyph_similarity(observed, template) for observed in observed_masks))
        scores.append((best, label))
    scores.sort(reverse=True)
    if not scores:
        return None, 0.0, 0.0
    best_score, label = scores[0]
    margin = best_score - (scores[1][0] if len(scores) > 1 else 0.0)
    # Closed-loop digits 6 and 8 can be nearly identical under tolerant
    # template matching. Eight is vertically and horizontally balanced, while
    # six has a characteristically sparse upper-right quadrant.
    if label in ("6", "8"):
        observed = observed_masks[1]
        half_y, half_x = observed.shape[0] // 2, observed.shape[1] // 2
        quadrants = [
            observed[:half_y, :half_x].mean(), observed[:half_y, half_x:].mean(),
            observed[half_y:, :half_x].mean(), observed[half_y:, half_x:].mean(),
        ]
        if max(quadrants) - min(quadrants) < 0.035:
            label = "8"
        elif quadrants[0] - quadrants[1] > 0.075:
            label = "6"
    if best_score < float(settings.get("template_score_min", 0.66)) or margin < float(settings.get("template_margin_min", 0.008)):
        return None, best_score, margin
    return label, best_score, margin


def recognize_continuation_hexagons(
    image: Image.Image,
    backend: OCRBackend,
    work: Path,
    page_number: int,
    base_items: list[OCRItem],
    config: dict | None = None,
    dpi: int = DEFAULT_DPI,
) -> list[OCRItem]:
    settings = (config or {}).get("continuation_hexagons", {})
    detections = detect_continuation_hexagons(image, config, dpi)
    if not detections:
        return []
    variants = list(settings.get("preprocess_variants", ["original", "autocontrast", "threshold"]))
    threshold_level = int(settings.get("threshold_level", 190))
    valid_pattern = re.compile(rf"^(?:{settings.get('valid_label_regex', '(?:[2-9]|1[0-9])[A-Z]?')})$")
    slot_width, slot_height, columns = 190, 130, 6
    slot_count = len(detections) * len(variants)
    rows = math.ceil(slot_count / columns)
    sheet = Image.new("RGB", (columns * slot_width, rows * slot_height), "white")
    slots = []
    for detection_index, detection in enumerate(detections):
        x0, y0, x1, y1 = detection["bbox_px"]
        inner = settings.get("inner_crop", [0.18, 0.1, 0.82, 0.9])
        crop = image.crop((
            int(x0 + (x1 - x0) * float(inner[0])),
            int(y0 + (y1 - y0) * float(inner[1])),
            int(x0 + (x1 - x0) * float(inner[2])),
            int(y0 + (y1 - y0) * float(inner[3])),
        ))
        for variant_index, variant in enumerate(variants):
            slot_index = detection_index * len(variants) + variant_index
            prepared = preprocess_ocr_crop(crop, variant, threshold_level)
            scale = min(5.0, 108 / max(prepared.width, prepared.height))
            content = prepared.resize((max(1, int(prepared.width * scale)), max(1, int(prepared.height * scale))))
            sx = (slot_index % columns) * slot_width + (slot_width - content.width) // 2
            sy = (slot_index // columns) * slot_height + (slot_height - content.height) // 2
            sheet.paste(content, (sx, sy))
            slots.append((sx, sy, sx + content.width, sy + content.height, detection_index, variant))
    recognized = recognize_crop(sheet, backend, work / f"continuations-{page_number}.png")
    observations: dict[int, list[dict]] = {index: [] for index in range(len(detections))}
    label_votes: dict[int, list[tuple[str, float]]] = {index: [] for index in range(len(detections))}
    for sx0, sy0, sx1, sy1, detection_index, variant in slots:
        pieces = [item for item in recognized if sx0 <= item.cx <= sx1 and sy0 <= item.cy <= sy1]
        pieces.sort(key=lambda item: (item.top, item.left))
        raw_text = "".join(item.text for item in pieces)
        cleaned = re.sub(r"[^A-Z0-9]", "", normalized_text(raw_text))
        confidence = sum(item.confidence for item in pieces) / len(pieces) if pieces else 0.0
        observations[detection_index].append({"variant": variant, "text": raw_text, "confidence": confidence})
        if valid_pattern.fullmatch(cleaned):
            label_votes[detection_index].append((cleaned, confidence))

    width, height = image.size
    margin = float(settings.get("exclude_border_margin", 0.035))
    output = []
    for detection_index, detection in enumerate(detections):
        x0, y0, x1, y1 = detection["bbox_px"]
        if x0 / width < margin or x1 / width > 1 - margin or y0 / height < margin or y1 / height > 1 - margin:
            continue
        # A page-level OCR read inside the polygon is useful as one additional
        # vote, but the three crop variants remain the primary evidence.
        for item in base_items:
            if x0 / width <= item.cx <= x1 / width and y0 / height <= item.cy <= y1 / height:
                cleaned = re.sub(r"[^A-Z0-9]", "", normalized_text(item.text))
                if valid_pattern.fullmatch(cleaned):
                    label_votes[detection_index].append((cleaned, item.confidence))
        ocr_vote_count = len(label_votes[detection_index])
        template_inner = settings.get("template_inner_crop", [0.18, 0.1, 0.82, 0.9])
        template_crop = image.crop((
            int(x0 + (x1 - x0) * float(template_inner[0])),
            int(y0 + (y1 - y0) * float(template_inner[1])),
            int(x0 + (x1 - x0) * float(template_inner[2])),
            int(y0 + (y1 - y0) * float(template_inner[3])),
        ))
        template_label, template_score, template_margin = classify_short_code(template_crop, settings)
        if template_label and valid_pattern.fullmatch(template_label):
            # Two votes let a confident shape read resolve a single noisy OCR
            # hypothesis, while three agreeing OCR variants still win.
            label_votes[detection_index].extend([(template_label, template_score)] * 2)
        label, agreement, ocr_confidence = majority_vote(label_votes[detection_index])
        if not label:
            continue
        metadata = {
            "source": "continuation_hexagon",
            "geometry_score": detection["geometry_score"],
            "vote_agreement": round(agreement, 4),
            "template_label": template_label,
            "template_score": round(template_score, 4),
            "template_margin": round(template_margin, 4),
            "ocr_vote_count": ocr_vote_count,
            "ocr_variants": observations[detection_index],
        }
        output.append(OCRItem(
            label, ocr_confidence,
            x0 / width, y0 / height, x1 / width, y1 / height, [metadata],
        ))
    return output


def cluster_positions(values: Iterable[int], tolerance: int = 7) -> list[int]:
    result: list[list[int]] = []
    for value in sorted(values):
        if result and value - int(np.median(result[-1])) <= tolerance:
            result[-1].append(value)
        else:
            result.append([value])
    return [int(np.median(group)) for group in result]


def detect_tables(image: Image.Image) -> list[dict]:
    mask = dark_mask(image)
    height, width = mask.shape
    horizontal = line_segments(mask, "horizontal", max(80, width // 80))
    vertical = line_segments(mask, "vertical", max(55, height // 100))
    candidates = []
    # Repeated horizontal rules with nearly identical endpoints define a grid body.
    endpoint_groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for line in horizontal:
        y, x0, x1 = line
        key = (round(x0 / 12), round(x1 / 12))
        endpoint_groups.setdefault(key, []).append(line)
    for lines in endpoint_groups.values():
        lines.sort()
        chains: list[list[tuple[int, int, int]]] = []
        for line in lines:
            if chains and line[0] - chains[-1][-1][0] <= max(160, height // 20):
                chains[-1].append(line)
            else:
                chains.append([line])
        for chain in chains:
            if len(chain) < 3:
                continue
            x0 = min(x[1] for x in chain)
            x1 = max(x[2] for x in chain)
            y0, y1 = chain[0][0], chain[-1][0]
            crossing = [v for v in vertical if x0 - 12 <= v[0] <= x1 + 12 and v[1] <= y0 + 12 and v[2] >= y1 - 12]
            if len(cluster_positions(v[0] for v in crossing)) < 3:
                continue
            # Include a partial super-header immediately above the detected body.
            row_gap = np.median(np.diff([x[0] for x in chain])) if len(chain) > 1 else 50
            for hy, hx0, hx1 in horizontal:
                overlap = max(0, min(x1, hx1) - max(x0, hx0))
                if 0 < y0 - hy <= row_gap * 1.7 and overlap >= .55 * min(x1 - x0, hx1 - hx0):
                    y0, x0, x1 = hy, min(x0, hx0), max(x1, hx1)
            candidates.append((x0, y0, x1, y1))
    # Merge duplicate/nested candidates.
    merged = []
    for box in sorted(candidates, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        x0, y0, x1, y1 = box
        if any(x0 >= a - 15 and y0 >= b - 15 and x1 <= c + 15 and y1 <= d + 15 for a, b, c, d in merged):
            continue
        merged.append(box)
    output = []
    for x0, y0, x1, y1 in merged:
        if x1 - x0 < width * .045 or y1 - y0 < height * .018:
            continue
        ys = cluster_positions(y for y, a, b in horizontal if y0 - 8 <= y <= y1 + 8 and max(0, min(x1, b) - max(x0, a)) >= .5 * (x1 - x0))
        xs = cluster_positions(x for x, a, b in vertical if x0 - 8 <= x <= x1 + 8 and max(0, min(y1, b) - max(y0, a)) >= .25 * (y1 - y0))
        xs = [x for x in xs if x0 - 8 <= x <= x1 + 8]
        if len(xs) >= 3 and len(ys) >= 3:
            output.append({"bbox_px": [min(xs), min(ys), max(xs), max(ys)], "x_lines": xs, "y_lines": ys})
    return output


def recognize_crop(image: Image.Image, backend: OCRBackend, path: Path) -> list[OCRItem]:
    image.save(path)
    width, height = image.size
    output = []
    for raw in backend.recognize(path):
        left = float(raw["x"]) * width
        top = float(raw.get("y_top", 1 - float(raw.get("y", 0)) - float(raw["height"]))) * height
        output.append(OCRItem(
            raw["text"], float(raw.get("confidence", 0)), left, top,
            left + float(raw["width"]) * width, top + float(raw["height"]) * height,
            raw.get("alternatives", []),
        ))
    return output


def clean_cell(text: str) -> str:
    text = normalized_text(text).replace("°", "").replace("*", "*")
    text = re.sub(r"\s+", " ", text).strip(" ·")
    return text


def logical_cells(mask: np.ndarray, xs: list[int], ys: list[int]) -> list[tuple[int, int, int, int, int, int]]:
    cells = []
    for row, (y0, y1) in enumerate(zip(ys, ys[1:])):
        start_col = 0
        for col in range(1, len(xs) - 1):
            x = xs[col]
            margin = max(2, int((y1 - y0) * .18))
            band = mask[y0 + margin:y1 - margin or y1, max(0, x - 2):x + 3]
            # A real rule is nearly continuous; glyph strokes can cross the probe
            # but occupy much less of the row height.
            occupied = np.any(band, axis=1) if band.size else np.array([], dtype=bool)
            runs = one_dimensional_runs(occupied, max_gap=1)
            longest = max((end - start + 1 for start, end in runs), default=0)
            boundary = bool(occupied.size and longest >= occupied.size * .62)
            if boundary:
                cells.append((row, start_col, col - start_col, xs[start_col], y0, x, y1))
                start_col = col
        cells.append((row, start_col, len(xs) - 1 - start_col, xs[start_col], y0, xs[-1], y1))
    return cells


HEADER_VOCABULARY = (
    "EQUIP. NO", "VALVE. NO", "INSTR. NO", "PUMP NO.", "TOP DRAIN",
    "BOTTOM DRAIN", "VIBRATION MONITOR INSTRUMENT TAG *", "A",
)


def canonical_header(text: str) -> str:
    if not text:
        return text
    key = re.sub(r"[^A-Z]", "", normalized_text(text))
    scored = []
    for value in HEADER_VOCABULARY:
        target = re.sub(r"[^A-Z]", "", value)
        scored.append((difflib.SequenceMatcher(None, key, target).ratio(), value))
    score, value = max(scored)
    return value if score >= .56 else text


def choose_candidate(candidates: list[str], pattern: str) -> str | None:
    for candidate in candidates:
        value = clean_cell(candidate)
        if re.fullmatch(pattern, value):
            return value
    return None


def normalize_table(matrix: list[list[str]], cell_candidates: list[list[list[str]]], cells: list[tuple], cell_objects: list[dict]) -> None:
    if not matrix:
        return
    # Canonicalize recognized header language without altering unknown headers.
    header_rows = 2 if len(matrix) > 1 and any(re.fullmatch(r"\d+", value) for value in matrix[0] if value) else 1
    for row in range(header_rows):
        for col, value in enumerate(matrix[row]):
            if not re.fullmatch(r"\d+", value or ""):
                matrix[row][col] = canonical_header(value)
    # Numbered super-headers often lose a single narrow glyph; complete only an
    # unambiguous 1..N sequence.
    if header_rows == 2:
        numeric_cols = [(col, int(value)) for col, value in enumerate(matrix[0]) if value.isdigit()]
        if numeric_cols:
            last_col, last_value = numeric_cols[-1]
            start_col = last_col - last_value + 1
            if start_col >= 0 and all(not matrix[0][start_col + n - 1] or matrix[0][start_col + n - 1] == str(n) for n in range(1, last_value + 1)):
                for n in range(1, last_value + 1):
                    matrix[0][start_col + n - 1] = str(n)

    # Map merged header cells onto their atomic columns.
    atomic_headers = ["" for _ in matrix[0]]
    for obj in cell_objects:
        if obj["row"] >= header_rows:
            continue
        value = matrix[obj["row"]][obj["column"]]
        if not value or value.isdigit():
            continue
        for col in range(obj["column"], obj["column"] + obj["column_span"]):
            atomic_headers[col] = (atomic_headers[col] + " " + value).strip()

    data_start = header_rows
    for row in range(data_start, len(matrix)):
        for col, value in enumerate(matrix[row]):
            if not value:
                continue
            candidates = cell_candidates[row][col] or [value]
            header = atomic_headers[col]
            if "PUMP" in header:
                chosen = choose_candidate(candidates, r"P-\d+[A-D]")
                value = chosen or value
                value = re.sub(r"^P-(\d+)(8)$", r"P-\1B", value)
                value = re.sub(r"^P-(\d+)(4)$", r"P-\1A", value)
            elif "VALVE" in header or "DRAIN" in header or header == "A":
                chosen = choose_candidate(candidates, r"(?:\d-SW-)?V\d+")
                value = chosen or value
                if re.fullmatch(r"[YV]?\d{3,4}", value):
                    value = "V" + re.sub(r"\D", "", value)
                value = re.sub(r"^V(\d+)E$", r"V\g<1>0", value)
            elif "INSTR" in header and "VIBRATION" not in header:
                chosen = choose_candidate(candidates, r"\d-SW-LG-\d+[A-Z]")
                value = chosen or value
                if re.fullmatch(r"\d-SW-LG-\d+8", value):
                    value = value[:-1] + "B"
            elif "VIBRATION" in header:
                chosen = choose_candidate(candidates, r"\d{5}")
                value = chosen or value
            matrix[row][col] = value

    # Pump identifiers commonly form A/B/C/D banks. Resolve OCR's B/8 and D/0
    # confusions from the local sequence rather than a sheet-specific tag list.
    pump_cols = [i for i, value in enumerate(atomic_headers) if "PUMP" in value]
    for col in pump_cols:
        values = [matrix[row][col] for row in range(data_start, len(matrix))]
        for offset in range(len(values) - 3):
            group = values[offset:offset + 4]
            first = re.fullmatch(r"P-(\d+)A", group[0] or "")
            third = re.fullmatch(r"P-(\d+)C", group[2] or "")
            if first and third and first.group(1) == third.group(1):
                base = first.group(1)
                matrix[data_start + offset + 1][col] = f"P-{base}B"
                matrix[data_start + offset + 3][col] = f"P-{base}D"
        # Adjacent A/B pair with one digit confused in the numeric base.
        for row in range(data_start, len(matrix) - 1):
            a = re.fullmatch(r"P-(\d+)A", matrix[row][col] or "")
            b = re.fullmatch(r"P-(\d+)B", matrix[row + 1][col] or "")
            if a and b and len(a.group(1)) == len(b.group(1)):
                distance = sum(x != y for x, y in zip(a.group(1), b.group(1)))
                if distance == 1:
                    matrix[row][col] = f"P-{b.group(1)}A"

    # Top/bottom drain valve pairs are sequential in a pump row. This catches a
    # single ambiguous digit only when both observed values already nearly agree.
    top_cols = [i for i, value in enumerate(atomic_headers) if value == "TOP DRAIN"]
    bottom_cols = [i for i, value in enumerate(atomic_headers) if value == "BOTTOM DRAIN"]
    if len(top_cols) == len(bottom_cols) == 1:
        tc, bc = top_cols[0], bottom_cols[0]
        for row in range(data_start, len(matrix)):
            top_match = re.fullmatch(r"V(\d+)", matrix[row][tc] or "")
            bottom_match = re.fullmatch(r"V(\d+)", matrix[row][bc] or "")
            if top_match and bottom_match:
                top_number, bottom_number = int(top_match.group(1)), int(bottom_match.group(1))
                if bottom_number != top_number + 1 and abs(bottom_number - top_number) <= 12:
                    matrix[row][bc] = f"V{top_number + 1}"

    # Repair numeric instrument columns when a strong +1 sequence is present.
    for col, header in enumerate(atomic_headers):
        if "VIBRATION" not in header:
            continue
        observed = []
        for row in range(data_start, len(matrix)):
            match = re.fullmatch(r"\d{5}", matrix[row][col] or "")
            if match:
                observed.append((row, int(match.group(0))))
        bases = [value - (row - data_start) for row, value in observed]
        if len(bases) >= 4:
            base = max(set(bases), key=bases.count)
            if bases.count(base) >= 3:
                for row, _ in observed:
                    matrix[row][col] = str(base + row - data_start)

    # Reflect normalized strings back into the cell metadata.
    for obj in cell_objects:
        obj["text"] = matrix[obj["row"]][obj["column"]]


def extract_table(image: Image.Image, spec: dict, backend: OCRBackend, work: Path, index: int, page_number: int) -> dict | None:
    x0, y0, x1, y1 = spec["bbox_px"]
    xs = spec["x_lines"]
    ys = spec["y_lines"]
    mask = dark_mask(image)
    cells = logical_cells(mask, xs, ys)
    slots = []
    slot_width, slot_height, padding = 420, 130, 24
    columns = 4
    rows = math.ceil(len(cells) / columns)
    sheet = Image.new("RGB", (columns * slot_width, rows * slot_height), "white")
    for slot, cell in enumerate(cells):
        _, _, _, cx0, cy0, cx1, cy1 = cell
        inset = 5
        crop = image.crop((cx0 + inset, cy0 + inset, cx1 - inset, cy1 - inset)).convert("RGB")
        scale = min((slot_width - 2 * padding) / max(1, crop.width), (slot_height - 2 * padding) / max(1, crop.height))
        resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        sx = (slot % columns) * slot_width + padding
        sy = (slot // columns) * slot_height + padding
        sheet.paste(resized, (sx, sy))
        slots.append((sx, sy, sx + resized.width, sy + resized.height))
    recognized = recognize_crop(sheet, backend, work / f"contact-{page_number}-{index}.png")
    cell_texts = []
    cell_candidate_texts = []
    for slot in slots:
        sx0, sy0, sx1, sy1 = slot
        pieces = [item for item in recognized if sx0 <= item.cx <= sx1 and sy0 <= item.cy <= sy1]
        pieces.sort(key=lambda item: (item.top, item.left))
        cell_texts.append(clean_cell(" ".join(item.text for item in pieces)))
        alternatives = []
        if len(pieces) == 1:
            alternatives = [alt.get("text", "") for alt in pieces[0].alternatives]
        cell_candidate_texts.append(uniq([cell_texts[-1]] + [clean_cell(value) for value in alternatives]))
    cell_objects = []
    matrix = [["" for _ in range(len(xs) - 1)] for _ in range(len(ys) - 1)]
    candidate_matrix = [[[] for _ in range(len(xs) - 1)] for _ in range(len(ys) - 1)]
    for cell, text, candidates in zip(cells, cell_texts, cell_candidate_texts):
        row, col, col_span, *_ = cell
        matrix[row][col] = text
        candidate_matrix[row][col] = candidates
        cell_objects.append({"row": row, "column": col, "row_span": 1, "column_span": col_span, "text": text})
    normalize_table(matrix, candidate_matrix, cells, cell_objects)
    combined = " ".join(cell_texts)
    title_block = "REV" in combined and "DATE" in combined and ("DSGN" in combined or "DESCRIPTION" in combined)
    nonempty_body_rows = sum(any(cell for cell in row) for row in matrix[1:])
    if title_block or nonempty_body_rows < 2:
        return None
    width, height = image.size
    return {
        "table_id": f"page_{page_number}_table_{index}",
        "page": page_number,
        "bbox": [round(x0 / width, 5), round(y0 / height, 5), round(x1 / width, 5), round(y1 / height, 5)],
        "rows": matrix,
        "cells": cell_objects,
    }


def write_tables(tables: list[dict], output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(exist_ok=True)
    for table in tables:
        with (table_dir / f"{table['table_id']}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(table["rows"])


def process(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    dpi = int(args.dpi or config.get("render", {}).get("dpi", DEFAULT_DPI))
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="engineering_extract_") as temp:
        work = Path(temp)
        pages = render_input(input_path, work, dpi)
        backend = OCRBackend(work, args.ocr_backend)
        documents = []
        confidence_documents = []
        review_queue = []
        all_tables = []
        raw_pages = []
        for page_number, page_path in enumerate(pages, 1):
            print(f"page {page_number}/{len(pages)}: OCR", file=sys.stderr, flush=True)
            ocr = recognize_tiled(page_path, backend, work, config)
            profile_name = select_document_profile(ocr, config, args.profile)
            profile = profile_settings(config, profile_name)
            print(f"page {page_number}: profile={profile_name}", file=sys.stderr, flush=True)
            image = Image.open(page_path).convert("RGB")
            bubble_ocr = (
                recognize_instrument_bubbles(image, backend, work, page_number, ocr, config, dpi, profile_name)
                if profile.get("instrument_bubbles", True) else []
            )
            continuation_ocr = (
                recognize_continuation_hexagons(image, backend, work, page_number, ocr, config, dpi)
                if profile.get("continuation_hexagons", True) else []
            )
            dedup_distance = float(config.get("ocr", {}).get("dedup_distance", 0.018))
            ocr = deduplicate_ocr(ocr + bubble_ocr + continuation_ocr, dedup_distance)
            raw_pages.append({"page": page_number, "profile": profile_name, "items": [asdict(item) for item in ocr]})
            answers = extract_pid(ocr, config, profile_name)
            document_id = args.document_id + page_number - 1
            documents.append({"document_id": document_id, "answers": answers})
            confidence_report, queued = build_confidence_report(document_id, page_number, answers, ocr, config)
            confidence_report["profile"] = profile_name
            confidence_documents.append(confidence_report)
            review_queue.extend(queued)
            if profile_name == "unknown_pid":
                review_queue.insert(0, {
                    "document_id": document_id,
                    "page": page_number,
                    "category": "document_profile",
                    "value": profile_name,
                    "confidence": 0.0,
                    "review_threshold": 1.0,
                    "source": "profile_router",
                    "bbox": None,
                    "evidence_text": None,
                    "reasons": [
                        "no_supported_document_family_matched",
                        "full_page_review_required",
                    ],
                })
            if args.mode in ("auto", "table"):
                specs = detect_tables(image)
                print(f"page {page_number}: {len(specs)} grid candidate(s)", file=sys.stderr, flush=True)
                for index, spec in enumerate(specs, 1):
                    table = extract_table(image, spec, backend, work, index, page_number)
                    if table:
                        all_tables.append(table)
        (output_dir / "structured_output.json").write_text(json.dumps({"documents": documents}, indent=2) + "\n", encoding="utf-8")
        (output_dir / "extraction_metadata.json").write_text(json.dumps({"documents": confidence_documents}, indent=2) + "\n", encoding="utf-8")
        (output_dir / "review_queue.json").write_text(json.dumps({"items": review_queue}, indent=2) + "\n", encoding="utf-8")
        (output_dir / "raw_ocr.json").write_text(json.dumps({"pages": raw_pages}, indent=2) + "\n", encoding="utf-8")
        (output_dir / "tables.json").write_text(json.dumps({"tables": all_tables}, indent=2) + "\n", encoding="utf-8")
        write_tables(all_tables, output_dir)
    print(f"wrote {output_dir / 'structured_output.json'}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="PDF or image to process")
    parser.add_argument("--output", default="output/run", help="output directory")
    parser.add_argument("--document-id", type=int, default=1, help="ID assigned to the first page")
    parser.add_argument("--mode", choices=("auto", "pid", "table"), default="auto")
    parser.add_argument("--ocr-backend", choices=("auto", "vision", "tesseract"), default="auto")
    parser.add_argument(
        "--profile", choices=("auto", "unknown_pid", "legacy_nuclear_pid", "generic_isa_pid"), default="auto",
        help="document-family grammar; auto selects from OCR evidence",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="JSON file containing detector thresholds and confidence policy")
    parser.add_argument("--dpi", type=int, default=None, help="override render.dpi from the config file")
    return parser.parse_args()


if __name__ == "__main__":
    process(parse_args())
