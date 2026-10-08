from pathlib import Path

from src.migration import PipelineConfig, run_pipeline


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = run_pipeline(PipelineConfig(
        input_csv=root / "data" / "Project_1" / "Starrating" / "alldata_2025jul.csv",
        output_dir=root / "outputs",
    ))
    print(f"Processed {len(result['analysis'])} hospitals and {len(result['included_measures'])} measures.")
