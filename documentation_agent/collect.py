"""Load SAS source, macros, and the CSV mapping for one program."""

from __future__ import annotations

from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class ProgramContext:
    program_id: str
    sas_file: str
    sas_code: str
    macros: str
    outputs: tuple[str, ...]
    mapping: str


def parse_mapping(text: str, heading: str) -> str:
    """Return the mapping block that starts with `heading` (e.g. 'Program 0:')."""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith(heading):
            start = index
            break
    if start is None:
        raise ValueError(f"No mapping section starting with {heading!r}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("Program ") and index != start:
            end = index
            break
    block = "\n".join(lines[start:end]).strip()
    if not block:
        raise ValueError(f"Empty mapping section for {heading!r}")
    return block


def collect(program_id: str) -> ProgramContext:
    program = config.get_program(program_id)
    if not program.path.is_file():
        raise FileNotFoundError(f"SAS program not found: {program.path}")
    if not config.MACROS_FILE.is_file():
        raise FileNotFoundError(f"Macros file not found: {config.MACROS_FILE}")
    if not config.MAPPING_FILE.is_file():
        raise FileNotFoundError(f"Mapping file not found: {config.MAPPING_FILE}")

    mapping_text = config.MAPPING_FILE.read_text(encoding="utf-8", errors="replace")
    return ProgramContext(
        program_id=program.id,
        sas_file=program.sas_file,
        sas_code=program.path.read_text(encoding="utf-8", errors="replace"),
        macros=config.MACROS_FILE.read_text(encoding="utf-8", errors="replace"),
        outputs=tuple(program.outputs),
        mapping=parse_mapping(mapping_text, program.mapping_heading),
    )
