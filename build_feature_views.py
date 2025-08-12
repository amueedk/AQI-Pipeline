"""
Build (or get) Feature Views for online inference.

Creates two Feature Views keyed by `time_str`:
- historic_fv → from feature group `aqi_clean_features_v2` v1
- forecasts_fv → from feature group `aqi_weather_forecasts` v1

Notes:
- Uses SDK patterns compatible with your environment (get_feature_view/create_feature_view, no unsupported args).
- Optional quick validation can be enabled via env VALIDATE=1 to fetch a few keys from the online store using get_feature_vectors().
"""

import os
import logging
from typing import Optional, Tuple

import hopsworks

from config import HOPSWORKS_CONFIG


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_feature_views")


def login_project() -> Tuple[object, object]:
    api_key = os.getenv("HOPSWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not set in environment")
    project_name = HOPSWORKS_CONFIG.get("project_name", "")
    if not project_name:
        raise RuntimeError("HOPSWORKS_CONFIG['project_name'] missing")
    logger.info(f"Logging into Hopsworks project: {project_name}...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    logger.info("Connected to Hopsworks Feature Store")
    return project, fs


def get_or_create_fv(fs, fv_name: str, version: int, query, description: str):
    """Retrieve FV if it exists (handling None-returning SDKs) else create it."""
    try:
        fv = fs.get_feature_view(name=fv_name, version=version)
        if fv is not None:
            logger.info(f"Feature View exists: {fv_name} v{version}")
            return fv
        else:
            logger.info(f"Feature View not found (None). Will create: {fv_name} v{version}")
    except Exception:
        logger.info(f"Feature View lookup raised. Will create: {fv_name} v{version}")

    # Create when missing
    fv = fs.create_feature_view(
        name=fv_name,
        version=version,
        description=description,
        labels=[],
        query=query,
    )
    logger.info(f"Created Feature View: {fv_name} v{version}")
    return fv


def build_feature_views(validate: bool = False) -> bool:
    try:
        _, fs = login_project()

        # Historic FV from aqi_clean_features_v2
        hist_fg_name = "aqi_clean_features_v2"
        hist_fv_name = "historic_fv"
        logger.info(f"Preparing historic Feature View from FG '{hist_fg_name}'...")
        hist_fg = fs.get_feature_group(name=hist_fg_name, version=1)
        hist_query = hist_fg.select_all()
        hist_fv = get_or_create_fv(
            fs,
            fv_name=hist_fv_name,
            version=1,
            query=hist_query,
            description="Historic engineered AQI features keyed by time_str",
        )

        # Forecasts FV from aqi_weather_forecasts
        fc_fg_name = "aqi_weather_forecasts"
        fc_fv_name = "forecasts_fv"
        logger.info(f"Preparing forecasts Feature View from FG '{fc_fg_name}'...")
        fc_fg = fs.get_feature_group(name=fc_fg_name, version=1)
        fc_query = fc_fg.select_all()
        fc_fv = get_or_create_fv(
            fs,
            fv_name=fc_fv_name,
            version=1,
            query=fc_query,
            description="72h weather+pollution forecasts keyed by time_str for inference",
        )

        # Ensure a training dataset exists capturing all features in the query
        for fv, nm in [(hist_fv, hist_fv_name), (fc_fv, fc_fv_name)]:
            try:
                # Try to create a fresh training dataset version capturing current schema
                td = fv.create_training_dataset(
                    description=f"Auto TD for {nm} with full feature schema",
                    data_format="parquet",
                )
                logger.info("Created training dataset for %s: version=%s", nm, getattr(td, 'version', 'n/a'))
                # Initialize serving to this TD version so online vectors use full schema
                try:
                    # Newer SDK signature
                    fv.init_serving(training_dataset_version=getattr(td, 'version', None))
                except TypeError:
                    try:
                        # Older SDK may accept version as positional
                        fv.init_serving(getattr(td, 'version', None))
                    except Exception:
                        fv.init_serving()
                logger.info("Initialized serving for %s", nm)
            except Exception as e:
                logger.info("Training dataset creation skipped/failed for %s: %s", nm, e)

        if validate:
            logger.info("VALIDATE=1 → Quick online fetch sanity check (small sample)")
            # Fallback: get a small set of keys from offline and query online FV
            try:
                import pandas as pd  # local import to avoid dependency if not validating
                # Read minimal offline sample to obtain time_str keys
                df_hist = hist_fg.read()
                df_fc = fc_fg.read()
                keys_hist = df_hist["time_str"].dropna().astype(str).tail(3).tolist()
                keys_fc = df_fc["time_str"].dropna().astype(str).head(3).tolist()
                logger.info(f"Sampling hist keys: {keys_hist}")
                logger.info(f"Sampling fc keys: {keys_fc}")
                # Use get_feature_vectors (compatible with your SDK) to query online
                try:
                    _ = hist_fv.get_feature_vectors(keys_hist)
                    logger.info("historic_fv online query OK (get_feature_vectors)")
                except Exception as e:
                    logger.warning(f"historic_fv online query failed: {e}")
                try:
                    _ = fc_fv.get_feature_vectors(keys_fc)
                    logger.info("forecasts_fv online query OK (get_feature_vectors)")
                except Exception as e:
                    logger.warning(f"forecasts_fv online query failed: {e}")
            except Exception as e:
                logger.warning(f"Validation step skipped/failed: {e}")

        # Final existence check and summary
        hist_fv_check = fs.get_feature_view(name=hist_fv_name, version=1)
        fc_fv_check = fs.get_feature_view(name=fc_fv_name, version=1)
        logger.info(
            "✅ Feature Views ready: %s v%s (exists=%s), %s v%s (exists=%s)",
            hist_fv_name, 1, hist_fv_check is not None, fc_fv_name, 1, fc_fv_check is not None
        )
        return True
    except Exception as e:
        logger.error(f"Failed to build Feature Views: {e}")
        return False


def main():
    validate = os.getenv("VALIDATE", "0") == "1"
    ok = build_feature_views(validate=validate)
    print("✅ Done" if ok else "❌ Failed")


if __name__ == "__main__":
    main()


