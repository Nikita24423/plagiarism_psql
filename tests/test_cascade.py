from cascade import (
    classify_chunks,
    EXACT,
    PARAPHRASE,
    SEMANTIC,
    classify_match,
    content_tokens,
    paraphrase_similarity,
)


def test_content_tokens_drop_stopwords_and_short_words():
    tokens = content_tokens("И в работе рассмотрены методы поиска")
    assert "и" not in tokens
    assert "в" not in tokens
    assert "работе" in tokens
    assert "методы" in tokens


def test_paraphrase_similarity_is_high_for_shared_content_words():
    a = "методы поиска похожих документов в корпусе университета"
    b = "методы поиска похожих документов применяются в корпусе университета"
    assert paraphrase_similarity(a, b) > 0.15


def test_paraphrase_similarity_is_low_for_unrelated_texts():
    assert paraphrase_similarity("сортировка массива пузырьком", "рецепт борща со сметаной") < 0.15


def test_classify_match_cascade_exact_then_paraphrase_then_semantic():
    assert classify_match(0.9, 0.0, lexical_threshold=0.6) == EXACT
    assert classify_match(0.2, 0.4, lexical_threshold=0.6) == PARAPHRASE
    assert classify_match(0.1, 0.05, lexical_threshold=0.6) == SEMANTIC


def test_fragment_is_exact_for_lexical_hit():
    from cascade import make_fragment

    query = "методы поиска похожих документов в корпусе университета подробно"
    source = "методы поиска похожих документов в корпусе университета подробно"
    frag = make_fragment(query, source, 0.91, "lexical", lexical_threshold=0.6)
    assert frag["match_type"] == EXACT
    assert frag["query_text"] == query
    assert frag["source_text"] == source


def test_fragment_keeps_whole_chunk_text():
    # Кусок поиска — до 500 символов; раньше обрезался до 400, и подсветка
    # не покрывала конец каждого куска (полная копия давала ~90%).
    from cascade import make_fragment

    chunk = "а" * 500
    frag = make_fragment(chunk, chunk, 1.0, "lexical", lexical_threshold=0.6)
    assert frag["query_text"] == chunk
    assert frag["source_text"] == chunk


def test_fragments_are_not_capped_per_source():
    # Регресс: при потолке 20 фрагментов на источник отчёт по подсветке
    # застывал на ~27% при любой доле заимствования выше ~20%.
    from cascade import append_fragment

    bag = []
    for i in range(60):
        append_fragment(bag, {"query_text": str(i), "source_text": "", "score": 0.9, "match_type": SEMANTIC})
    assert len(bag) == 60


def test_one_fragment_per_chunk_upgraded_to_exact():
    # dense- и lexical-сигнал одного куска дают один фрагмент, а не два.
    from cascade import append_fragment

    bag = []
    append_fragment(bag, {"query_text": "кусок", "source_text": "src", "score": 0.95, "match_type": SEMANTIC})
    append_fragment(bag, {"query_text": "кусок", "source_text": "src", "score": 1.0, "match_type": EXACT})
    append_fragment(bag, {"query_text": "кусок", "source_text": "src", "score": 0.9, "match_type": SEMANTIC})
    assert len(bag) == 1
    assert bag[0]["match_type"] == EXACT
    assert bag[0]["score"] == 1.0


def test_fragment_is_semantic_when_words_do_not_overlap():
    from cascade import make_fragment

    frag = make_fragment(
        "сортировка массива пузырьком на практике",
        "рецепт борща со сметаной и чесноком",
        0.82,
        "semantic",
        lexical_threshold=0.6,
    )
    assert frag["match_type"] == SEMANTIC


def test_bigram_coverage_joins_neighbouring_source_chunks():
    # Скопированный кусок лёг на стык двух кусков источника: с каждым по
    # отдельности совпадает половина, вместе — почти всё.
    from cascade import bigram_coverage

    src1 = "Система мониторинга нагрузки Prometheus выводит графики через Grafana и хранит метрики"
    src2 = "Балансировщик перераспределяет запросы к поддоменам на конкретные приложения по работе с каналами"
    query = ("выводит графики через Grafana и хранит метрики. "
             "Балансировщик перераспределяет запросы к поддоменам на конкретные приложения")
    assert bigram_coverage(query, [src1]) < 0.6
    assert bigram_coverage(query, [src2]) < 0.6
    assert bigram_coverage(query, [src1, src2]) >= 0.9


def test_bigram_coverage_is_low_for_unrelated_text_on_same_topic():
    from cascade import bigram_coverage

    query = "Беспилотный летательный аппарат измеряет концентрацию вредных газов около предприятия"
    sources = ["Чат-бот торговой компании отвечает клиентам и сохраняет сообщения в базу данных",
               "Аппарат передаёт данные о качестве воздуха оператору по радиоканалу"]
    assert bigram_coverage(query, sources) < 0.2
    assert bigram_coverage("", sources) == 0.0


# --- Шаблонные фразы (classify_chunks) ---

def test_phrase_shared_by_many_works_is_template():
    # Каждый кусок резюме похож на разные резюме из базы, ни одно не совпадает по 3 кускам.
    chunks = [{"r1", "r2", "r3", "r4", "r5"}, {"r6", "r7", "r8", "r9", "r10"}, {"r2", "r11", "r12", "r13", "r14"}]
    flags, strong = classify_chunks(chunks, min_sources=5, strong_min_chunks=3)
    assert flags == [True, True, True]
    assert strong == set()


def test_copy_of_one_work_is_not_template_even_if_popular():
    # Работу A уже скопировали многие: каждый кусок похож на 6 работ,
    # но A (и копии) совпадают по всем кускам — это списывание.
    copies = {"A", "c1", "c2", "c3", "c4", "c5"}
    flags, strong = classify_chunks([copies, copies, copies], min_sources=5, strong_min_chunks=3)
    assert flags == [False, False, False]
    assert "A" in strong


def test_single_source_match_is_plagiarism():
    flags, _ = classify_chunks([{"etalon"}, {"etalon"}, set()], min_sources=5, strong_min_chunks=3)
    assert flags == [False, False, False]


def test_template_chunk_next_to_real_copy():
    # Кусок 0 — шапка вуза (много работ), куски 1–3 — копия работы E.
    chunks = [{"u1", "u2", "u3", "u4", "u5"}, {"E"}, {"E"}, {"E"}]
    flags, strong = classify_chunks(chunks, min_sources=5, strong_min_chunks=3)
    assert flags == [True, False, False, False]
    assert strong == {"E"}


def test_only_near_verbatim_matches_make_a_source_strong():
    # Резюме r1 похоже на проверяемое по смыслу во всех кусках (типовые фразы),
    # но дословно не совпадает ни в одном — оно не «сильное», куски остаются шаблоном.
    many = {"r1", "r2", "r3", "r4", "r5"}
    flags, strong = classify_chunks([many, many, many], 5, 3, chunk_verbatim_sources=[set(), set(), set()])
    assert flags == [True, True, True]
    assert strong == set()
    # Если те же куски совпали с r1 дословно — это копия r1.
    flags, strong = classify_chunks([many, many, many], 5, 3, chunk_verbatim_sources=[{"r1"}, {"r1"}, {"r1"}])
    assert flags == [False, False, False]
    assert strong == {"r1"}


def test_rule_can_be_disabled():
    flags, _ = classify_chunks([{"a", "b", "c", "d", "e", "f"}], min_sources=0)
    assert flags == [False]
