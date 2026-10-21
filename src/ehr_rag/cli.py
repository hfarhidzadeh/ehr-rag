"""Command line interface: ``python -m ehr_rag <command>``.

Commands: ``generate``, ``ingest``, ``ask``, ``eval``, ``app``. Paths come from ``EHR_RAG_*`` environment
variables (see :mod:`ehr_rag.config`) or the global ``--data-dir`` / ``--index-dir`` options.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from ehr_rag.config import PROJECT_ROOT, Settings, load_settings
from ehr_rag.types import RETRIEVAL_MODES

logger = logging.getLogger(__name__)

APP_SCRIPT = PROJECT_ROOT / "app" / "streamlit_app.py"
EMBEDDERS = ("minilm", "tfidf")
LLMS = ("auto", "claude", "extractive")


# --------------------------------------------------------------------------- settings


def _settings(args: argparse.Namespace, **overrides) -> Settings:
    paths = {
        "data_dir": Path(args.data_dir).expanduser() if getattr(args, "data_dir", None) else None,
        "index_dir": Path(args.index_dir).expanduser() if getattr(args, "index_dir", None) else None,
        "reports_dir": Path(args.reports_dir).expanduser() if getattr(args, "reports_dir", None) else None,
    }
    return load_settings(**paths, **overrides)


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- commands


def cmd_generate(args: argparse.Namespace) -> int:
    from ehr_rag.synth.generator import generate_dataset

    settings = _settings(args)
    n = args.n_patients if args.n_patients is not None else settings.n_patients
    seed = args.seed if args.seed is not None else settings.seed
    dataset = generate_dataset(n_patients=n, seed=seed)
    dataset.write(settings.data_dir)
    print(f"Wrote synthetic dataset: {len(dataset.patients)} patients, {len(dataset.notes)} notes, "
          f"{len(dataset.labs)} lab rows (seed={seed}) -> {settings.data_dir}")
    print("Next: python -m ehr_rag ingest --embedder tfidf   (or --embedder minilm)")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from ehr_rag.ingest.phi_guard import NonSyntheticDataError
    from ehr_rag.pipeline import EHRRagPipeline, PHIDetectedError

    settings = _settings(args, embedder=args.embedder)
    try:
        pipe = EHRRagPipeline.build_index(settings, generator=_NoGenerator())
    except FileNotFoundError as exc:
        _err(str(exc))
        return 1
    except (NonSyntheticDataError, PHIDetectedError) as exc:
        _err(str(exc))
        return 3
    m = pipe.manifest
    print(f"Indexed {m['n_docs']} documents -> {m['n_chunks']} chunks for {m['n_patients']} patients "
          f"(embedder={m['embedder']}, dim={m['dim']}) at {settings.embedder_index_dir}")
    return 0


class _NoGenerator:
    """Placeholder so ``ingest`` never builds a Claude client."""

    name = "none"

    def generate(self, question, patient_id, contexts):  # pragma: no cover - never called
        raise RuntimeError("ingest does not generate answers")


def _load_pipeline(settings: Settings):
    from ehr_rag.pipeline import EHRRagPipeline, IndexNotBuiltError

    try:
        return EHRRagPipeline.load(settings)
    except IndexNotBuiltError as exc:
        _err(str(exc))
        return None


def _print_answer(answer) -> None:
    print(f"Patient {answer.patient_id} | Q: {answer.question}")
    print()
    print(answer.text)
    print()
    if answer.citations:
        print("Citations:")
        for c in answer.citations:
            print(f"  [{c.label}] {c.title} ({c.date}) - {c.doc_id} [{c.chunk_id}]")
    else:
        print("Citations: none")
    print(f"Abstained: {'yes' if answer.abstained else 'no'}")
    model = f", model={answer.model}" if answer.model else ""
    print(f"Generator: {answer.mode}{model} | retrieved {len(answer.retrieved)} chunks | "
          f"{answer.latency_ms:.0f} ms")
    if answer.warnings:
        print("Warnings:")
        for w in answer.warnings:
            print(f"  - {w}")


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args, embedder=args.embedder, llm=args.llm, retrieval_mode=args.mode, top_k=args.k)
    pipe = _load_pipeline(settings)
    if pipe is None:
        return 1
    patient_id = args.patient.strip().upper()
    try:
        answer = pipe.ask(args.question, patient_id, doc_types=args.doc_type or None)
    except ValueError as exc:
        _err(str(exc))
        return 2
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
        return 0
    _print_answer(answer)
    if args.show_retrieved:
        print("Retrieved:")
        for r in answer.retrieved:
            print(f"  {r.rank:>2}. {r.score:.4f} {r.chunk.chunk_id} | {r.chunk.title} | {r.chunk.date}"
                  f"{' | ' + r.chunk.section if r.chunk.section else ''}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    try:
        runner = importlib.import_module("ehr_rag.evaluation.runner")  # honours sys.modules (tests, stubs)
        from ehr_rag.evaluation.questions import build_questions
    except ImportError as exc:
        _err(f"the evaluation module is not available ({exc}). "
             "ehr_rag.evaluation.runner / questions must be implemented before `eval` can run.")
        return 2
    from ehr_rag.synth.generator import SyntheticDataset

    settings = _settings(args, embedder=args.embedder, llm=args.llm, retrieval_mode=args.mode, top_k=args.k)
    pipe = _load_pipeline(settings)
    if pipe is None:
        return 1
    dataset = SyntheticDataset.load(settings.data_dir)
    questions = build_questions(dataset, pipe.chunks, seed=settings.seed)
    if args.vocab_expansions:
        from ehr_rag.generation import ExtractiveGenerator

        pipe.generator = ExtractiveGenerator(term_expansions=runner.vocab_term_expansions())
    llm = getattr(pipe.generator, "name", settings.llm)
    if args.vocab_expansions:
        llm = f"{llm}-vocab"
    if args.compare:
        report = runner.compare_retrievers(pipe, questions, k=settings.top_k, limit=args.limit)
        name = f"compare_{settings.embedder}"
    else:
        report = runner.run_eval(pipe, questions, k=settings.top_k, mode=settings.retrieval_mode,
                                 limit=args.limit, generate=not args.retrieval_only)
        name = f"eval_{settings.embedder}_{settings.retrieval_mode}_{llm if not args.retrieval_only else 'none'}"
    json_path, md_path = runner.write_report(report, Path(settings.reports_dir), name)
    n = len(questions) if args.limit is None else min(args.limit, len(questions))
    print(f"Evaluated {n} questions -> {json_path} and {md_path}")
    _print_eval_summary(report)
    return 0


def _print_eval_summary(report: dict) -> None:
    keys = ("recall@1", "recall@k", "primary_recall@k", "mrr", "answer_accuracy", "false_abstention_rate",
            "correct_abstention_rate", "citation_validity", "leakage_rate", "latency_mean_ms")
    if report.get("kind") == "compare":
        groups = [(m, s["by_control"]["positive"], s["overall"]) for m, s in report["modes"].items()]
        print("mode      " + "  ".join(f"{k:>16}" for k in ("recall@1", "recall@k", "mrr", "verbatim R@1",
                                                             "leakage_rate")))
        for m, pos, overall in groups:
            verb = report["modes"][m]["by_category"].get("verbatim", {}).get("recall@1")
            vals = (pos.get("recall@1"), pos.get("recall@k"), pos.get("mrr"), verb, overall.get("leakage_rate"))
            print(f"{m:<10}" + "  ".join(f"{v:>16.3f}" if isinstance(v, float) else f"{'-':>16}" for v in vals))
    else:
        overall = report["summary"]["overall"]
        for key in keys:
            value = overall.get(key)
            if value is not None:
                print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    for c in report.get("controls", []):
        value = f"{c['value']:.3f}" if isinstance(c.get("value"), float) else c.get("value")
        print(f"  [{c['status']}] {c['name']}: {value}")


def cmd_app(args: argparse.Namespace) -> int:
    script = Path(APP_SCRIPT)
    if not script.exists():
        _err(f"{script} not found; the Streamlit app has not been written yet.")
        return 1
    try:
        target = str(script.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        target = str(script.resolve())
    cmd = [sys.executable, "-m", "streamlit", "run", target]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    env = dict(os.environ)
    for name in ("data_dir", "index_dir", "reports_dir"):
        value = getattr(args, name, None)
        if value:
            env[f"EHR_RAG_{name.upper()}"] = str(Path(value).expanduser().resolve())
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ehr_rag",
        description="EHR-RAG: retrieval-augmented QA over SYNTHETIC electronic health records.",
    )
    parser.add_argument("--data-dir", help="synthetic dataset directory (default: EHR_RAG_DATA_DIR or data/synthetic)")
    parser.add_argument("--index-dir", help="index root directory (default: EHR_RAG_INDEX_DIR or data/index)")
    parser.add_argument("--reports-dir", help="evaluation reports directory (default: reports)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for INFO logs, -vv for DEBUG")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("generate", help="generate the synthetic dataset")
    p.add_argument("--n-patients", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("ingest", help="chunk, embed and index the synthetic documents")
    p.add_argument("--embedder", choices=EMBEDDERS, default=None)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question about one patient")
    p.add_argument("question")
    p.add_argument("--patient", "-p", required=True, help="patient id, e.g. P001")
    p.add_argument("--mode", choices=RETRIEVAL_MODES, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--llm", choices=LLMS, default=None)
    p.add_argument("--embedder", choices=EMBEDDERS, default=None)
    p.add_argument("--doc-type", action="append", help="restrict retrieval to a doc type (repeatable)")
    p.add_argument("--show-retrieved", action="store_true", help="also list the retrieved chunks")
    p.add_argument("--json", action="store_true", help="print the Answer as JSON")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="run the gold-question evaluation")
    p.add_argument("--embedder", choices=EMBEDDERS, default=None)
    p.add_argument("--mode", choices=RETRIEVAL_MODES, default=None)
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--llm", choices=LLMS, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--compare", action="store_true", help="compare random/bm25/dense/hybrid retrievers")
    p.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    p.add_argument("--vocab-expansions", action="store_true",
                   help="extractive generator with lay-term -> abbreviation expansions from the synthetic vocab "
                        "(a separate configuration: it uses the vocabulary the semantic questions are built from)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("app", help="run the Streamlit app")
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_app)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING if args.verbose == 0 else (logging.INFO if args.verbose == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
