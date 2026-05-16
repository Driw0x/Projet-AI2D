import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    """
    Lance tous les fichiers de test du dossier tests/.
    """
    result = pytest.main([
        str(ROOT_DIR / "tests"),
        "-v",
    ])

    if result == 0:
        print("\nTous les tests sont passés.")
    else:
        print("\nCertains tests ont échoué.")

    return result


if __name__ == "__main__":
    raise SystemExit(main())
