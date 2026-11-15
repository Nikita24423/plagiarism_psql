import base64
import io

import pytest

from text_normalize import extract_text, normalize_text


def test_two_format_versions_collapse_to_same_text():
    # «одна работа» из PDF и из DOCX: разные пробелы/переносы -> один текст.
    pdf_like = "Цель  работы.\r\n\r\n\r\nАнализ сайтов   выполнен."
    docx_like = "Цель работы.\n\nАнализ сайтов выполнен."
    assert normalize_text(pdf_like) == normalize_text(docx_like)


def test_nonbreaking_and_narrow_spaces_become_regular():
    assert normalize_text("а б в г") == "а б в г"


def test_soft_hyphen_and_zero_width_removed():
    assert normalize_text("пере­нос​слова") == "переносслова"


def test_crlf_normalized():
    assert normalize_text("строка1\r\nстрока2\rстрока3") == "строка1\nстрока2\nстрока3"


def test_runs_of_blank_lines_collapse():
    assert normalize_text("а\n\n\n\n\nб") == "а\n\nб"


def test_trailing_spaces_stripped():
    assert normalize_text("слово   \nдругое") == "слово\nдругое"


def test_nfkc_folds_compatibility_forms():
    # римская цифра Ⅳ (U+2163) -> IV при NFKC
    assert normalize_text("глава Ⅳ") == "глава IV"


def test_empty_and_blank_safe():
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  ") == ""


def test_meaningful_content_preserved():
    t = "Краулинговый бюджет это количество страниц, которые робот сканирует."
    assert normalize_text(t) == t


def test_determinism_same_input_same_output():
    t = "Любой   текст\r\n\r\nс разными пробелами."
    assert normalize_text(t) == normalize_text(t)


# --- Извлечение текста из файла (extract_text) ---


def _as_content(ext: str, data: bytes) -> str:
    return f"FILE_BASE64|{ext}|" + base64.b64encode(data).decode()


def _docx_bytes(build) -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    build(document)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def test_extract_plain_text_passes_through():
    assert extract_text("обычный текст работы") == "обычный текст работы"


def test_docx_keeps_tables_in_document_order():
    def build(d):
        d.add_paragraph("Введение работы.")
        table = d.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Метод"
        table.cell(0, 1).text = "Сложность"
        table.cell(1, 0).text = "Быстрая сортировка"
        table.cell(1, 1).text = "O(n log n)"
        d.add_paragraph("Заключение работы.")

    text = extract_text(_as_content("docx", _docx_bytes(build)))
    lines = text.splitlines()
    assert lines[0] == "Введение работы."
    assert "Быстрая сортировка | O(n log n)" in lines
    assert lines[-1] == "Заключение работы."
    assert lines.index("Быстрая сортировка | O(n log n)") < lines.index("Заключение работы.")


def test_docx_merged_cells_not_duplicated():
    def build(d):
        table = d.add_table(rows=1, cols=3)
        merged = table.cell(0, 0).merge(table.cell(0, 1))
        merged.text = "Объединённая"
        table.cell(0, 2).text = "Отдельная"

    text = extract_text(_as_content("docx", _docx_bytes(build)))
    assert text == "Объединённая | Отдельная"


def test_doc_format_gives_clear_error():
    with pytest.raises(ValueError, match=r"\.doc"):
        extract_text(_as_content("doc", b"\xd0\xcf\x11\xe0 old word"))


def test_unknown_format_is_rejected_not_analyzed_as_text():
    with pytest.raises(ValueError, match="не поддерживается"):
        extract_text(_as_content("rtf", b"{\\rtf1 text}"))


def test_scanned_pdf_without_text_layer_is_rejected():
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ValueError, match="текстового слоя"):
        extract_text(_as_content("pdf", buf.getvalue()))


def test_broken_file_gives_value_error():
    with pytest.raises(ValueError, match="Не удалось извлечь"):
        extract_text(_as_content("docx", b"not a zip"))
