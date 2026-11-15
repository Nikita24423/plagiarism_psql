import asyncio
import threading

import job_store
from job_store import JobStore, run_queued_job
from university_queues import UniversityQueues
from ai_result import build_ai_fragments, resolve_ai_percent
from ai_scoring import load_ai_config, sigmoid_ai_percent


def test_resolve_null_when_disabled_or_empty(monkeypatch):
    monkeypatch.delenv("AI_DETECT_D0", raising=False)
    monkeypatch.delenv("AI_DETECT_SCALE", raising=False)
    cfg = load_ai_config()
    assert resolve_ai_percent(enabled=False, scores=[1.0], config=cfg) is None
    assert resolve_ai_percent(enabled=True, scores=[], config=cfg) is None


def test_resolve_percent_when_scores_present(monkeypatch):
    monkeypatch.setenv("AI_DETECT_D0", "1.0")
    monkeypatch.setenv("AI_DETECT_SCALE", "1.5")
    cfg = load_ai_config()
    got = resolve_ai_percent(enabled=True, scores=[0.0, 0.0], config=cfg)
    assert got == sigmoid_ai_percent(0.0, 1.0, 1.5)


def test_build_ai_fragments_fields_and_limit(monkeypatch):
    monkeypatch.setenv("AI_DETECT_D0", "1.0")
    monkeypatch.setenv("AI_DETECT_SCALE", "1.5")
    cfg = load_ai_config()
    pairs = [("один" * 100, 0.0), ("два", 2.0), ("три", -1.0)]
    fr = build_ai_fragments(pairs, cfg, max_fragments=2, preview_chars=10)
    assert len(fr) == 2
    assert fr[0]["text"] == ("один" * 100)[:10]
    assert fr[0]["d_score"] == 0.0
    assert fr[0]["ai_probability"] == sigmoid_ai_percent(0.0, 1.0, 1.5)
    assert fr[1]["text"] == "два"


def test_build_ai_fragments_zero_limit_returns_empty(monkeypatch):
    cfg = load_ai_config()
    assert build_ai_fragments([("x", 1.0)], cfg, max_fragments=0) == []


# --- Статус задачи /v1/jobs: queued, пока ждёт в очереди; processing — только при старте ---


def test_job_stays_queued_until_scheduler_starts_it(monkeypatch):
    store = JobStore()
    monkeypatch.setattr(job_store, "_store", store)

    async def scenario():
        queues = UniversityQueues(concurrency=1)
        queues.start()
        release_first = threading.Event()

        first = await store.create("bsuir", 1, "first.txt")
        second = await store.create("bsuir", 2, "second.txt")

        def submit(job_id, fn):
            return lambda: queues.submit("bsuir", fn, on_start=lambda: store.mark_processing(job_id))

        t1 = asyncio.create_task(run_queued_job(first.job_id, submit(first.job_id, lambda: (release_first.wait(5), {"n": 1})[1])))
        t2 = asyncio.create_task(run_queued_job(second.job_id, submit(second.job_id, lambda: {"n": 2})))

        # Первая задача занимает единственный слот, вторая ждёт в очереди.
        for _ in range(100):
            if (await store.get(first.job_id)).status == "processing":
                break
            await asyncio.sleep(0.01)
        assert (await store.get(first.job_id)).status == "processing"
        assert (await store.get(second.job_id)).status == "queued"

        release_first.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), 5)
        assert (await store.get(first.job_id)).status == "completed"
        assert (await store.get(second.job_id)).status == "completed"
        assert (await store.get(second.job_id)).result == {"n": 2}
        await queues.stop()

    asyncio.run(scenario())
