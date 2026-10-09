"""VS Code friendly entry point; identical to python -m mycelia."""
from mycelia.cli import main

if __name__ == '__main__':
    raise SystemExit(main())

