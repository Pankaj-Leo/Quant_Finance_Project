import yfinance as yf
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Config loaded ✅")
    print(cfg)

    # Create data folder
    save_path = Path(cfg.data.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="fetch_yfinance"):
        all_data = []
        for t in cfg.data.tickers:
            print(f"Downloading {t}...")
            df = yf.download(t, start=cfg.data.start_date, end=cfg.data.end_date, progress=False)
            df["Ticker"] = t
            all_data.append(df)

        full_df = pd.concat(all_data)
        out_file = save_path / "ohlcv.parquet"
        full_df.to_parquet(out_file)

        print(f"✅ Saved OHLCV data to {out_file}")
        mlflow.log_param("tickers", cfg.data.tickers)
        mlflow.log_artifact(str(out_file))

if __name__ == "__main__":
    main()
