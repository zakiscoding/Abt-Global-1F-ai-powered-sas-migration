from documentation_agent.collect import collect, parse_mapping
from documentation_agent.config import get_program


def test_program0_resolves_to_standardization_sas():
    program = get_program("program0")
    assert program.path.is_file()
    assert program.sas_file == "0 - Data and Measure Standardization_2025Jul.sas"
    assert program.path.name == program.sas_file


def test_program0_collects_keep_hos_and_expected_outputs():
    ctx = collect("program0")
    assert "%keep_hos" in ctx.sas_code
    assert "%macro keep_hos" in ctx.macros
    assert ctx.outputs == (
        "STD_DATA_2025JUL_ANALYSIS",
        "MEASURE_AVERAGE_STDDEV_2025JUL",
        "LESS100_MEASURE",
    )
    for name in ctx.outputs:
        assert name in ctx.mapping or f"{name}.csv" in ctx.mapping


def test_parse_mapping_returns_program0_block_only():
    text = (
        "SAS Program to CSV Output Mapping\n\n"
        "Program 0: Data and Measure Standardization\n"
        "- LESS100_MEASURE.csv: dropped measures.\n"
        "- MEASURE_AVERAGE_STDDEV_2025JUL.csv: mean and std.\n"
        "- STD_DATA_2025JUL_ANALYSIS.csv: standardized scores.\n\n"
        "Program 1: First Stage\n"
        "- OUTCOME_MORTALITY.csv: mortality domain.\n"
    )
    block = parse_mapping(text, "Program 0:")
    assert "LESS100_MEASURE.csv" in block
    assert "STD_DATA_2025JUL_ANALYSIS.csv" in block
    assert "OUTCOME_MORTALITY" not in block
