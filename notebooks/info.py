For a solid, production-grade experiment tracking workflow using a feature factory with multiple feature sets, the best approach is modular, reproducible, and extensible. Here’s how pros typically do it:

✅ TL;DR Best Practice Summary
Use CLI arguments to select feature sets (--config basic, --config behavioural)

Have separate config YAML/JSON files for each feature set

Baseline features should be included in all configs as shared logic, not conditionally stacked

Use a unified run_pipeline.py script that takes the config as input and logs to MLflow/W&B

Log feature set name, hyperparameters, model version, and metrics in experiment tracking

Keep feature engineering modular using a factory pattern or strategy injection

Avoid boolean flags like USE_BEHAVIOURAL=True inside a single config — that’s brittle and doesn’t scale for more than 2 variants.

🔧 Project Structure Recommendation
project/
├── run_pipeline.py
├── configs/
│ ├── baseline.yaml
│ ├── behavioural.yaml
│ └── temporal.yaml
├── src/
│ ├── ingest_data.py
│ ├── data_cleaning.py
│ ├── feature_factory/
│ │ ├── __init__.py
│ │ ├── baseline.py
│ │ ├── behavioural.py
│ │ └── temporal.py
│ └── model_training.py
└── logs/

How to Structure the CLI


Your run_pipeline.py might look like:

python run_pipeline.py --config baseline
python run_pipeline.py --config behavioural

Or with full path:

python run_pipeline.py --config configs/baseline.yaml

Feature Factory Strategy Pattern (recommended)


You define feature logic like this:

class FeatureEngineer(ABC):
@abstractmethod
def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
pass

class BaselineFeatures(FeatureEngineer):
def generate_features(self, df):
# Add common features
return df

class BehaviouralFeatures(FeatureEngineer):
def generate_features(self, df):
df = BaselineFeatures().generate_features(df)
# Add behavioural rolling features
return df

Then in your pipeline:

feature_strategy = FeatureFactory.create(config["feature_strategy"])
df = feature_strategy.generate_features(df)


Experiment Tracking with MLflow/W&B


Every run should log:

Feature strategy name (e.g. baseline, behavioural)

Model type and version

Metrics (AUC, precision, etc.)

Dataset hash (optional)

Timestamp

Git commit hash (ideal for reproducibility)


Example:
import mlflow

mlflow.start_run(run_name=config["feature_strategy"])
mlflow.log_params(config)
mlflow.log_metrics(metrics)
mlflow.end_run()

What to Avoid
❌ Hardcoding feature toggles in one config file

❌ Running baseline then stacking behavioural features in one run — this creates coupling and bad tracking

❌ Boolean flags like USE_BEHAVIOURAL=True — doesn’t scale to 3+ strategies

✅ Final Recommendations

Goal

Recommendation

Compare feature strategies

One config file per strategy

Reproducible runs

CLI + config loading + MLflow

Feature engineering

Strategy pattern with clean inheritance

Scaling experiments

Use loops/schedulers to automate config sweeps

Deployment

Lock best config into production.yaml