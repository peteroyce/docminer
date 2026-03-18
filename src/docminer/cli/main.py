"""Click CLI entry point for DocMiner."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="docminer")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """DocMiner — document intelligence pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@cli.command("extract")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "csv", "markdown"]),
    default="json",
    show_default=True,
    help="Output format",
)
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out-file", type=click.Path(path_type=Path), default=None, help="Write output to file")
@click.pass_context
def extract_cmd(
    ctx: click.Context,
    file: Path,
    output: str,
    config: Path | None,
    out_file: Path | None,
) -> None:
    """Extract text, entities, and tables from FILE."""
    from docminer.config.loader import load_config
    from docminer.core.pipeline import Pipeline
    from docminer.output.formatter import OutputFormatter

    cfg = load_config(config)
    pipeline = Pipeline(config=cfg)

    try:
        result = pipeline.process_file(file)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    formatter = OutputFormatter()
    formatted = formatter.format(result, fmt=output)  # type: ignore[arg-type]

    if out_file:
        out_file.write_text(formatted, encoding="utf-8")
        click.echo(f"Output written to {out_file}")
    else:
        click.echo(formatted)


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------


@cli.command("classify")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None)
@click.pass_context
def classify_cmd(ctx: click.Context, file: Path, config: Path | None) -> None:
    """Classify the document type of FILE."""
    from docminer.classification.classifier import DocumentClassifier
    from docminer.config.loader import load_config
    from docminer.extraction import create_extractor
    from docminer.utils.file_utils import detect_file_type

    load_config(config)
    file_type = detect_file_type(file)
    extractor = create_extractor(file_type)

    try:
        document = extractor.extract_document(file)
    except Exception as exc:
        click.echo(f"Extraction error: {exc}", err=True)
        sys.exit(1)

    classifier = DocumentClassifier()
    result = classifier.classify(document)

    output = {
        "file": str(file),
        "document_type": result.document_type,
        "confidence": round(result.confidence, 4),
        "all_scores": {k: round(v, 4) for k, v in result.all_scores.items()},
    }
    click.echo(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@cli.command("analyze")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "csv", "markdown"]),
    default="json",
    show_default=True,
)
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--out-file", type=click.Path(path_type=Path), default=None)
@click.pass_context
def analyze_cmd(
    ctx: click.Context,
    file: Path,
    output: str,
    config: Path | None,
    out_file: Path | None,
) -> None:
    """Run the full pipeline on FILE."""
    # analyze is an alias for extract with all pipeline steps enabled
    ctx.invoke(extract_cmd, file=file, output=output, config=config, out_file=out_file)


# ---------------------------------------------------------------------------
# pipeline (batch)
# ---------------------------------------------------------------------------


@cli.command("pipeline")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive", "-r", is_flag=True, default=False, help="Process sub-directories")
@click.option("--pattern", default="*.pdf", show_default=True, help="Glob pattern for files")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["json", "csv", "markdown"]),
    default="json",
    show_default=True,
)
@click.option("--out-dir", type=click.Path(path_type=Path), default=None, help="Directory for output files")
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None)
@click.pass_context
def pipeline_cmd(
    ctx: click.Context,
    directory: Path,
    recursive: bool,
    pattern: str,
    output: str,
    out_dir: Path | None,
    config: Path | None,
) -> None:
    """Batch-process all matching files in DIRECTORY."""
    from docminer.config.loader import load_config
    from docminer.core.pipeline import Pipeline
    from docminer.output.formatter import OutputFormatter

    cfg = load_config(config)
    pipeline = Pipeline(config=cfg)
    formatter = OutputFormatter()

    results = pipeline.process_directory(directory, recursive=recursive, pattern=pattern)
    click.echo(f"Processed {len(results)} document(s)", err=True)

    for result in results:
        formatted = formatter.format(result, fmt=output)  # type: ignore[arg-type]
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(result.document.source_path).stem
            ext = {"json": ".json", "csv": ".csv", "markdown": ".md"}[output]
            out_path = out_dir / f"{stem}{ext}"
            out_path.write_text(formatted, encoding="utf-8")
        else:
            click.echo(formatted)
            click.echo("---")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", "-p", default=8000, show_default=True, type=int)
@click.option("--reload", is_flag=True, default=False)
@click.option("--config", "-c", type=click.Path(exists=True, path_type=Path), default=None)
@click.pass_context
def serve_cmd(ctx: click.Context, host: str, port: int, reload: bool, config: Path | None) -> None:
    """Start the DocMiner API server."""
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required to run the server: pip install uvicorn", err=True)
        sys.exit(1)

    from docminer.api.app import create_app
    from docminer.config.loader import load_config

    cfg = load_config(config)
    click.echo(f"Starting DocMiner API on http://{host}:{port}")

    if reload:
        uvicorn.run("docminer.api.app:create_app", host=host, port=port, reload=True)
    else:
        app = create_app(config=cfg)
        uvicorn.run(app, host=host, port=port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
