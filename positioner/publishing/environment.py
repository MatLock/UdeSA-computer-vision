"""Environment selection for Merchant Center targeting."""
from __future__ import annotations

import enum
import os


class Environment(enum.Enum):
    TEST = "test"
    PRODUCTION = "production"


def get_environment(env_str: str = None) -> Environment:
    """Resolve the target environment.

    Args:
        env_str: explicit environment string, or None to read MERCHANT_ENV.

    Returns:
        Environment enum value. Defaults to Environment.TEST when unset or empty.
    """
    if env_str is None:
        env_str = os.environ.get('MERCHANT_ENV')
    if not env_str:
        return Environment.TEST
    return Environment(env_str)
