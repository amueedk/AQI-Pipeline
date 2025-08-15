"""
Utilities to store and retrieve models in Hopsworks Model Registry,
with a robust fallback to Datasets storage for SDK compatibility.
"""

import os
import time
import logging
from typing import Dict, Optional

try:
    import hopsworks  # type: ignore
except Exception:  # pragma: no cover
    hopsworks = None  # type: ignore

from config import HOPSWORKS_CONFIG

logger = logging.getLogger(__name__)


def _login():
    if hopsworks is None:
        raise RuntimeError("hopsworks client not installed")
    api_key = os.getenv("HOPSWORKS_API_KEY", "")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is not set")
    project_name = HOPSWORKS_CONFIG.get("project_name", "")
    if not project_name:
        raise RuntimeError("HOPSWORKS_CONFIG['project_name'] is missing")
    logger.info(f"Logging into Hopsworks project: {project_name}...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    return project


def register_or_upload_model(
    artifacts_dir: str,
    model_name: str,
    metrics: Optional[Dict[str, float]] = None,
    description: str = "",
) -> bool:
    """
    Try to register a model in the Model Registry; if not available, upload to Datasets.
    Returns True on success.
    """
    project = _login()
    # First try model registry
    try:
        mr = project.get_model_registry()
        # Some SDKs expose mr.python.create_model; others have create_model directly
        creator = None
        if hasattr(mr, "python") and hasattr(mr.python, "create_model"):
            creator = mr.python.create_model
        elif hasattr(mr, "create_model"):
            creator = mr.create_model
        if creator is not None:
            logger.info("Registering model in Hopsworks Model Registry: %s", model_name)
            try:
                mdl = creator(
                    name=model_name,
                    metrics=metrics or {},
                    description=description or f"Registered model {model_name}",
                )
                # Save uploads the whole artifacts directory
                mdl.save(artifacts_dir)
                logger.info("Model registered successfully: %s", model_name)
                # Also sync a Production copy in Datasets for easy consumption
                try:
                    ds_api = _login().get_dataset_api()
                    prod_remote = f"/Models/{model_name}/Production"
                    logger.info("Syncing Production copy in Datasets: %s", prod_remote)
                    ds_api.upload(artifacts_dir, prod_remote, overwrite=True)
                except Exception as ee:
                    logger.warning("Datasets Production sync failed (non-fatal): %s", ee)
                return True
            except Exception as e:
                logger.warning("Model registry create/save failed, will fallback to Datasets: %s", e)
        else:
            logger.info("Model Registry create_model API not available; falling back to Datasets")
    except Exception as e:
        logger.warning("Model Registry not available; falling back to Datasets: %s", e)

    # Fallback: upload to Datasets
    try:
        ds_api = project.get_dataset_api()
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_remote = f"/Models/{model_name}/{ts}"
        logger.info("Uploading artifacts to Datasets: %s", base_remote)
        ds_api.upload(artifacts_dir, base_remote, overwrite=True)
        # Update/replace Production copy
        prod_remote = f"/Models/{model_name}/Production"
        logger.info("Syncing Production copy: %s", prod_remote)
        ds_api.upload(artifacts_dir, prod_remote, overwrite=True)
        logger.info("Artifacts uploaded to Datasets successfully")
        return True
    except Exception as e:
        logger.error("Failed to upload artifacts to Datasets: %s", e)
        return False


def download_production_artifacts(model_name: str, local_dir: str) -> bool:
    """
    Download the Production artifacts of a model from Datasets into local_dir.
    """
    project = _login()
    try:
        os.makedirs(local_dir, exist_ok=True)
        ds_api = project.get_dataset_api()
        remote_prod = f"/Models/{model_name}/Production"
        logger.info("Downloading Production artifacts from %s -> %s", remote_prod, local_dir)
        ds_api.download(remote_prod, local_dir, overwrite=True)
        return True
    except Exception as e:
        logger.error("Failed to download Production artifacts: %s", e)
        return False


def _get_latest_model(mr, model_name: str):
    """Return the latest version of a model, compatible with older SDKs."""
    # Always try to list versions first to get the actual latest
    try:
        # Some SDKs expose list_models(name=...) or get_models(name=...)
        if hasattr(mr, 'list_models'):
            models = mr.list_models(name=model_name)
        elif hasattr(mr, 'get_models'):
            models = mr.get_models(name=model_name)
        else:
            models = []
        
        # models could be list of dicts or objects with .version
        latest = None
        best_v = -1
        for m in models:
            v = getattr(m, 'version', None)
            if v is None and isinstance(m, dict):
                v = m.get('version')
            if isinstance(v, int) and v > best_v:
                latest = m
                best_v = v
        
        if latest is not None:
            # If latest is dict, get full model by name+version
            if isinstance(latest, dict):
                return mr.get_model(model_name, version=latest.get('version'))
            return latest
    except Exception as e:
        logger.warning(f"Failed to list models for {model_name}: {e}")
    
    # Fallback: try direct get_model without version (some SDKs return latest)
    try:
        mdl = mr.get_model(model_name)
        if mdl is not None:
            return mdl
    except Exception as e:
        logger.warning(f"Failed to get model {model_name} without version: {e}")
    
    raise RuntimeError(f"Model '{model_name}' not found in registry")


def download_registry_artifacts(model_name: str, local_dir: str, version: int | None = None) -> bool:
    """Download model artifacts from the Model Registry into local_dir."""
    project = _login()
    mr = project.get_model_registry()
    try:
        mdl = mr.get_model(model_name, version=version) if version is not None else _get_latest_model(mr, model_name)
        # Ensure a clean target directory to avoid overwrite errors on some SDKs
        import shutil
        if os.path.isdir(local_dir):
            try:
                shutil.rmtree(local_dir)
            except Exception:
                pass
        os.makedirs(local_dir, exist_ok=True)
        # Try common download method names
        for attr in ['download', 'download_model']:
            if hasattr(mdl, attr):
                # Some SDKs support overwrite kwarg; call with it if accepted
                try:
                    getattr(mdl, attr)(local_dir, overwrite=True)  # type: ignore[arg-type]
                except TypeError:
                    getattr(mdl, attr)(local_dir)
                return True
        # Some SDKs expose .get_artifact() returning a local path; copy to local_dir
        for attr in ['get_artifact', 'get_model_artifact']:
            if hasattr(mdl, attr):
                path = getattr(mdl, attr)()
                # If it's a directory, mirror into local_dir
                if os.path.isdir(path):
                    for entry in os.listdir(path):
                        src = os.path.join(path, entry)
                        dst = os.path.join(local_dir, entry)
                        if os.path.isdir(src):
                            shutil.copytree(src, dst, dirs_exist_ok=True)
                        else:
                            shutil.copy2(src, dst)
                    return True
        raise RuntimeError("Registry model object has no supported download method")
    except Exception as e:
        logger.error("Failed to download from Model Registry: %s", e)
        return False


