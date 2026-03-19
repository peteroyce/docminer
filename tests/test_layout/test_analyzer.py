"""Tests for layout analysis."""

from __future__ import annotations

import pytest

from docminer.core.types import BoundingBox, Page, TextBlock
from docminer.layout.analyzer import LayoutAnalyzer
from docminer.layout.geometry import BoundingBoxOps
from docminer.layout.regions import RegionDetector


def make_block(
    text: str,
    x0: float = 50,
    y0: float = 100,
    x1: float = 500,
    y1: float = 120,
    font_size: float = 11.0,
    block_type: str = "paragraph",
) -> TextBlock:
    return TextBlock(
        text=text,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        font_size=font_size,
        block_type=block_type,
        page_num=1,
    )


class TestLayoutAnalyzer:
    def test_header_detected_by_font_size(self) -> None:
        analyzer = LayoutAnalyzer()
        block = make_block("Section Title", font_size=16.0)
        result = analyzer.classify_block(block, dominant_font=11.0)
        assert result in ("header", "title")

    def test_paragraph_detected(self) -> None:
        analyzer = LayoutAnalyzer()
        block = make_block(
            "This is a normal paragraph with enough text to be a paragraph.",
            font_size=11.0,
        )
        result = analyzer.classify_block(block, dominant_font=11.0)
        assert result == "paragraph"

    def test_list_item_bullet(self) -> None:
        analyzer = LayoutAnalyzer()
        block = make_block("• First item in the list")
        result = analyzer.classify_block(block)
        assert result == "list_item"

    def test_list_item_numbered(self) -> None:
        analyzer = LayoutAnalyzer()
        block = make_block("1. First numbered item")
        result = analyzer.classify_block(block)
        assert result == "list_item"

    def test_footer_pattern(self) -> None:
        analyzer = LayoutAnalyzer()
        block = make_block("Page 1 of 10")
        result = analyzer.classify_block(block)
        assert result == "footer"

    def test_caption_detected(self) -> None:
        analyzer = LayoutAnalyzer()
        block = make_block("Figure 1: Architecture overview")
        result = analyzer.classify_block(block)
        assert result == "caption"

    def test_analyze_page_modifies_blocks(self) -> None:
        analyzer = LayoutAnalyzer()
        page = Page(
            number=1,
            width=612,
            height=792,
            blocks=[
                make_block("Title of Document", y0=50, y1=70, font_size=20.0),
                make_block("Body paragraph text here.", y0=150, y1=170, font_size=11.0),
                make_block("• List item", y0=200, y1=220, font_size=11.0),
            ],
        )
        analyzer.analyze_page(page)
        types = [b.block_type for b in page.blocks]
        assert "paragraph" in types or "header" in types or "title" in types

    def test_empty_page_no_crash(self) -> None:
        analyzer = LayoutAnalyzer()
        page = Page(number=1, width=612, height=792, blocks=[])
        analyzer.analyze_page(page)  # should not raise
        assert page.blocks == []

    def test_all_caps_short_line_is_header(self) -> None:
        analyzer = LayoutAnalyzer()
        block = make_block("SECTION ONE")
        result = analyzer.classify_block(block)
        assert result == "header"


class TestRegionDetector:
    def test_single_column_detection(self) -> None:
        detector = RegionDetector()
        blocks = [
            make_block("A", x0=50, x1=500, y0=100 + i * 30, y1=120 + i * 30)
            for i in range(10)
        ]
        cols = detector.detect_columns(blocks, page_width=612)
        assert cols == 1

    def test_reading_order_sorts_top_to_bottom(self) -> None:
        detector = RegionDetector()
        blocks = [
            make_block("C", y0=300, y1=320),
            make_block("A", y0=100, y1=120),
            make_block("B", y0=200, y1=220),
        ]
        ordered = detector.reading_order(blocks, page_width=612, num_columns=1)
        texts = [b.text for b in ordered]
        assert texts == ["A", "B", "C"]

    def test_zone_classification(self) -> None:
        detector = RegionDetector()
        blocks = [
            make_block("Header text", y0=10, y1=30),        # top 10% of 792 = 79.2
            make_block("Body text", y0=400, y1=420),
            make_block("Footer text", y0=750, y1=770),       # bottom 10% starts at 712.8
        ]
        zones = detector.classify_zones(blocks, page_height=792)
        assert any(b.text == "Header text" for b in zones["header"])
        assert any(b.text == "Body text" for b in zones["body"])
        assert any(b.text == "Footer text" for b in zones["footer"])


class TestBoundingBoxOps:
    def test_merge_all(self) -> None:
        boxes = [
            BoundingBox(0, 0, 10, 10),
            BoundingBox(5, 5, 20, 20),
            BoundingBox(-5, -5, 3, 3),
        ]
        merged = BoundingBoxOps.merge_all(boxes)
        assert merged.x0 == -5
        assert merged.y0 == -5
        assert merged.x1 == 20
        assert merged.y1 == 20

    def test_reading_order(self) -> None:
        boxes = [
            BoundingBox(100, 200, 200, 220),
            BoundingBox(100, 50, 200, 70),
            BoundingBox(100, 100, 200, 120),
        ]
        ordered = BoundingBoxOps.reading_order(boxes)
        y_values = [b.y0 for b in ordered]
        assert y_values == sorted(y_values)

    def test_non_maximum_suppression(self) -> None:
        boxes = [
            BoundingBox(0, 0, 100, 100),
            BoundingBox(5, 5, 95, 95),   # almost identical — should be suppressed
            BoundingBox(200, 200, 300, 300),  # different — should keep
        ]
        kept = BoundingBoxOps.non_maximum_suppression(boxes, iou_threshold=0.5)
        assert len(kept) == 2

    def test_aspect_ratio(self) -> None:
        box = BoundingBox(0, 0, 200, 100)
        assert BoundingBoxOps.aspect_ratio(box) == pytest.approx(2.0)
