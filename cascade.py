"""
Каскадная классификация заимствований — ЕДИНСТВЕННЫЙ источник истины.

Ранее типы заимствований (exact/paraphrase/semantic) считались отдельно
на TypeScript в guard-main и отдельно здесь — с разными порогами. Теперь
классификация выполняется только в этом модуле; веб-платформа использует
`match_type` из ответа ML-сервиса, а её локальный каскад — офлайн-фолбэк
на случай недоступности ML.

Каскад по источнику (первый сработавший тир определяет тип):
  1. exact      — лексический косинус ≥ LEXICAL_SCORE_THRESHOLD
                  (дословное копирование: совпали конкретные слова);
  2. paraphrase — совпадение 3-грамм по смысловым словам (Жаккар)
                  ≥ PARAPHRASE_THRESHOLD (переформулировка с сохранением
                  смысловой лексики);
  3. semantic   — только семантический сигнал dense-эмбеддингов
                  (глубокий рерайт: смысл совпал, слова — нет).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Hashable, Iterable, Sequence

# Стоп-слова: служебные слова несут мало сигнала о заимствовании,
# но сильно завышают сходство несвязанных текстов (ru + en).
STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все всё она так его но да ты к у же
    вы за бы по только ее её мне было вот от меня еще ещё нет о из ему когда
    даже ну ли если уже или ни быть был до вас вам ведь там потом себя может
    они тут где есть надо для мы тебя их чем была сам без чего раз тоже себе
    под будет тогда кто этот того потому этого какой ним здесь этом один мой
    тем чтобы нее неё были куда всех при об хотя тот через эти нас про всего
    них эту это той им более также
    the a an and or of to in on for with is are was were be been by at from
    as it its this that these those not no but if then than such can may
    """.split()
)

EXACT = "exact"
PARAPHRASE = "paraphrase"
SEMANTIC = "semantic"

# Порог Жаккара 3-грамм смысловых слов для тира «перефраз»
# (соответствует бывшему PARAPHRASE_THRESHOLD=15% в guard-main).
DEFAULT_PARAPHRASE_THRESHOLD = 0.15


def content_tokens(text: str) -> list[str]:
    """Смысловые слова: без пунктуации, коротких слов и стоп-слов."""
    return [
        t
        for t in re.findall(r"[a-zа-яё0-9]+", (text or "").lower())
        if len(t) > 2 and t not in STOPWORDS
    ]


def paraphrase_similarity(text1: str, text2: str, k: int = 3) -> float:
    """
    Оценка перефразирования: симметричный Жаккар k-грамм по смысловым словам.
    Ловит переписанный текст без дословных совпадений. Возвращает 0..1.
    """

    def build(tokens: list[str]) -> set[str]:
        if len(tokens) < k:
            return {" ".join(tokens)} if tokens else set()
        return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}

    a = build(content_tokens(text1))
    b = build(content_tokens(text2))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def _coverage_tokens(text: str) -> list[str]:
    # Те же токены, что у лексического вектора (worker._lexical_sparse).
    return [
        t for t in re.findall(r"[a-zа-яё0-9]+", (text or "").lower())
        if len(t) >= 2 and t not in STOPWORDS
    ]


def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = _coverage_tokens(text)
    return set(zip(tokens, tokens[1:]))


def bigram_coverage(query_text: str, source_texts) -> float:
    """
    Доля пар соседних смысловых слов куска работы, которые встречаются в
    кусках ОДНОГО источника вместе взятых. 0..1.

    Зачем: куски режутся по абзацам, и при переформатировании (другие
    абзацы, убраны заголовки/таблицы) скопированный кусок ложится на стык
    двух кусков источника. С каждым по отдельности сходство ниже порога,
    а вместе они покрывают кусок почти полностью (~0.95). У несвязанного
    текста покрытие ~0: случайно совпадают единичные слова, но не пары.
    """
    query = _bigrams(query_text)
    if not query:
        return 0.0
    covered: set[tuple[str, str]] = set()
    for text in source_texts:
        covered |= query & _bigrams(text)
    return len(covered) / len(query)


def classify_match(
    max_lexical_score: float,
    paraphrase_score: float,
    lexical_threshold: float,
    paraphrase_threshold: float = DEFAULT_PARAPHRASE_THRESHOLD,
) -> str:
    """Тип заимствования источника по каскаду exact → paraphrase → semantic."""
    if max_lexical_score >= lexical_threshold:
        return EXACT
    if paraphrase_score >= paraphrase_threshold:
        return PARAPHRASE
    return SEMANTIC


# Фрагменты — подсветка совпадений в отчёте (пары «кусок работы ↔ кусок
# источника»). Раньше их было не больше 20 на источник и каждый обрезался до
# 400 символов, а dense- и lexical-сигнал одного куска давали два фрагмента.
# Отчёт, построенный по подсветке, упирался в потолок ~10 кусков (~27% работы
# на 2400 слов) и не видел конец каждого куска. Теперь потолка нет: ровно один
# фрагмент на совпавший кусок и источник, текст куска целиком.
MAX_FRAGMENTS_PER_SOURCE: int | None = None
# Страховка от неожиданно длинных кусков: поиск режет текст по 500 символов.
FRAGMENT_TEXT_LIMIT = 1000


def classify_fragment(
    kind: str,
    score: float,
    query_text: str,
    source_text: str,
    lexical_threshold: float,
    paraphrase_threshold: float = DEFAULT_PARAPHRASE_THRESHOLD,
) -> str:
    """Тип одного фрагмента: лексический сигнал выше порога — дословно, иначе перефраз или смысл."""
    if kind == "lexical" and score >= lexical_threshold:
        return EXACT
    if paraphrase_similarity(query_text, source_text) >= paraphrase_threshold:
        return PARAPHRASE
    return SEMANTIC


def make_fragment(
    query_text: str,
    source_text: str,
    score: float,
    kind: str,
    lexical_threshold: float,
    paraphrase_threshold: float = DEFAULT_PARAPHRASE_THRESHOLD,
) -> dict:
    query = (query_text or "")[:FRAGMENT_TEXT_LIMIT]
    source = (source_text or "")[:FRAGMENT_TEXT_LIMIT]
    return {
        "query_text": query,
        "source_text": source,
        "score": round(float(score or 0.0), 4),
        "match_type": classify_fragment(
            kind, float(score or 0.0), query, source, lexical_threshold, paraphrase_threshold
        ),
    }


def append_fragment(
    fragments: list, fragment: dict, limit: int | None = MAX_FRAGMENTS_PER_SOURCE
) -> None:
    """Добавляет фрагмент к источнику: один фрагмент на кусок проверяемой работы.

    Повтор того же куска (второй сигнал — dense или lexical) не дублируется:
    фрагмент только повышается до `exact`, если новый сигнал дословный.
    """
    for existing in fragments:
        if existing["query_text"] == fragment["query_text"]:
            if fragment["match_type"] == EXACT and existing["match_type"] != EXACT:
                existing.update(fragment)
            return
    if limit is None or len(fragments) < limit:
        fragments.append(fragment)


# ---------------------------------------------------------------------------
# Шаблонные фразы
# ---------------------------------------------------------------------------
# Проблема: кусок работы считался заимствованным, если в базе нашёлся хотя бы
# один похожий кусок. В шаблонных документах (резюме, отчёты по одной
# методичке, титульные листы, «шапка» вуза) типовые фразы есть почти в каждой
# работе, и любой новый документ получал ~100%.
#
# Правило:
# - кусок, похожий на куски из `min_sources` и более РАЗНЫХ работ, — шаблон:
#   это общеупотребительный текст, а не копия конкретной работы;
# - исключение: если среди этих работ есть «сильный» источник — работа,
#   совпавшая с проверяемой хотя бы по `strong_min_chunks` кускам, — кусок
#   засчитывается как заимствование. Так ловится настоящее списывание, даже
#   если одну и ту же работу уже скопировали многие.
#
# Шаблонные куски не входят в plagiarism_percent и показываются отдельно
# (template_percent). Без зависимостей от PyTorch — покрыто тестами.

DEFAULT_TEMPLATE_MIN_SOURCES = 5
DEFAULT_STRONG_SOURCE_MIN_CHUNKS = 3
# «Почти дословное» совпадение куска — только оно делает источник сильным.
# Похожая по смыслу типовая фраза (dense 0.80–0.96) источник сильным не делает:
# иначе шаблонные документы, где многие фразы общие, взаимно «усиливают» друг
# друга и правило шаблона не срабатывает.
NEAR_VERBATIM_LEXICAL = 0.85
NEAR_VERBATIM_DENSE = 0.97
# Покрытие пар слов куска кусками источника (bigram_coverage):
# ≥ 0.9 — кусок целиком взят из этого источника, хоть и со сдвигом.
NEAR_VERBATIM_COVERAGE = 0.9


def classify_chunks(
    chunk_sources: Sequence[Iterable[Hashable]],
    min_sources: int = DEFAULT_TEMPLATE_MIN_SOURCES,
    strong_min_chunks: int = DEFAULT_STRONG_SOURCE_MIN_CHUNKS,
    chunk_verbatim_sources: Sequence[Iterable[Hashable]] | None = None,
) -> tuple[list[bool], set]:
    """
    chunk_sources[i] — работы-источники, похожие на i-й кусок (все кандидаты
    выше порога, а не только лучший).
    chunk_verbatim_sources[i] — те из них, с которыми i-й кусок совпал почти
    дословно; по ним считается «сила» источника. Не задано — используется
    chunk_sources (любое совпадение).

    Возвращает (is_template[i], strong_sources). min_sources <= 0 выключает
    правило: ни один кусок не считается шаблоном.
    """
    sources = [set(s) for s in chunk_sources]
    verbatim = sources if chunk_verbatim_sources is None else [set(s) for s in chunk_verbatim_sources]
    per_source = Counter(src for srcs in verbatim for src in srcs)
    strong = {src for src, n in per_source.items() if n >= max(1, strong_min_chunks)}
    if min_sources <= 0:
        return [False] * len(sources), strong
    flags = [len(srcs) >= min_sources and not (srcs & strong) for srcs in sources]
    return flags, strong
