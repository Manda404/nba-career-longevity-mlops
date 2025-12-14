from pathlib import Path

from nba_longevity.infrastructure.config.settings import load_infra_config
from nba_longevity.infrastructure.logging.logger import setup_logger
from nba_longevity.infrastructure.config.settings import InfraConfig

# Path infrastructure
PATH = "/Users/surelmanda/Downloads/nba-career-longevity-mlops/config/infra.yaml"

# Single source of truth
CONFIG_PATH_INFRA = Path(PATH)

# Load infrastructure configuration
config: InfraConfig = load_infra_config(CONFIG_PATH_INFRA)

# Initialize global logger ONCE
app_logger = setup_logger(config)
