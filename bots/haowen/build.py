"""Build the C++17 entry before the referee starts the game clock."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / ('bot.exe' if sys.platform == 'win32' else 'bot')
sys.exit(subprocess.call(['c++', '-std=c++17', '-O3', str(here / 'bot.cpp'), '-o', str(out)]))
