"""
Нормализация текста работы перед нарезкой на чанки.

Задача 10: одна и та же работа у двух студентов даёт разные показатели ИИ.
Детектор сам по себе детерминирован (eval, без сэмплинга), поэтому причина —
в тексте: разный формат (PDF vs DOCX), версия Word или пересохранение дают
слегка разный извлечённый текст (пробелы, переносы, неразрывные пробелы,
мягкие переносы), из-за чего сдвигаются границы чанков, а значит и
усреднённая статистика.

Нормализация сводит такие тривиальные различия к одному виду: Unicode NFKC,
единый перевод строки, схлопнутые пробелы, убранные служебные символы. Это не
делает PDF и DOCX байт-в-байт одинаковыми, но заметно уменьшает разброс
показателей между форматами одной работы. Применяется ДО нарезки, поэтому
одинаково влияет и на поиск заимствований, и на детектор ИИ.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
import unicodedata

# Пробельные символы, которые разные конвертеры вставляют по-разному:
# обычный пробел, таб, неразрывный пробел, узкие/типографские пробелы, BOM.
_HORIZONTAL_WS = re.compile(r"[ \t  -   　﻿]+")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_NEWLINES = re.compile(r"\n{3,}")

# Символы нулевой ширины и мягкий перенос — удаляются полностью.
_ZERO_WIDTH = dict.fromkeys(map(ord, "­​‌‍⁠"), None)


def normalize_text(text: str) -> str:
    """Приводит текст к стабильному виду, не меняя его смысл."""
    if not text:
        return text
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(_ZERO_WIDTH)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _HORIZONTAL_WS.sub(" ", t)
    t = _TRAILING_WS.sub("\n", t)
    t = _MANY_NEWLINES.sub("\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Извлечение текста из файла (до нормализации)
# ---------------------------------------------------------------------------
# Вход /v1/analyze и /v1/jobs: `content` — либо готовый текст, либо файл в виде
# `FILE_BASE64|<ext>|<base64>`. Без зависимостей от PyTorch — покрыто тестами.
#
# Поддерживается:
# - DOCX — абзацы и таблицы в порядке документа (раньше таблицы терялись);
# - PDF — текстовый слой страниц.
#
# Явная ошибка (ValueError → HTTP 400) вместо тихого неверного результата:
# - `.doc` (старый двоичный Word) — python-docx его не читает;
# - PDF без текстового слоя (скан) — раньше давал 0% заимствований;
# - неизвестный формат — раньше строка FILE_BASE64|… уходила в анализ как текст.

FILE_PREFIX = "FILE_BASE64|"
SUPPORTED = ("docx", "pdf")


def extract_text(content: str) -> str:
    if not isinstance(content, str) or not content.startswith(FILE_PREFIX):
        return content
    parts = content.split("|", 2)
    if len(parts) != 3:
        raise ValueError("Некорректный формат файла: ожидается FILE_BASE64|<расширение>|<base64>")
    _, ext, b64_data = parts
    ext = ext.strip().lower().lstrip(".")

    if ext == "doc":
        raise ValueError(
            "Формат .doc (Word 97–2003) не поддерживается. Сохраните работу как .docx или .pdf"
        )
    if ext not in SUPPORTED:
        raise ValueError(
            f"Формат .{ext} не поддерживается. Поддерживаются: " + ", ".join("." + e for e in SUPPORTED)
        )

    try:
        file_bytes = base64.b64decode(b64_data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Не удалось декодировать файл: {exc}") from exc

    try:
        if ext == "pdf":
            text = _pdf_text(file_bytes)
            if not text.strip():
                raise ValueError(
                    "В PDF нет текстового слоя (скан или изображение). "
                    "Загрузите PDF с текстом или файл .docx"
                )
            return text
        return _docx_text(file_bytes)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 — битый файл → 400, а не 500
        # Раньше вместо ошибки индексировалась строка-заглушка, и все битые
        # файлы «совпадали» с ней на 100%.
        raise ValueError(f"Не удалось извлечь текст из файла: {exc}") from exc


def _pdf_text(file_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _docx_text(file_bytes: bytes) -> str:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(file_bytes))
    parts: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            parts.append(Paragraph(child, document).text)
        elif tag == "tbl":
            parts.extend(_table_rows(Table(child, document)))
    return "\n".join(parts).strip()


def _table_rows(table) -> list[str]:
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = cell.text.strip()
            # Объединённые ячейки python-docx отдаёт повторно — пропускаем дубли.
            if text and (not cells or cells[-1] != text):
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return rows
