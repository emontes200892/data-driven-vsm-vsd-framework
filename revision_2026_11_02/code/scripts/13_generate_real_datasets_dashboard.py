"""Render audited figures; historical writer is preserved in original_scripts/."""
from pathlib import Path
import importlib.util
p=Path(__file__).with_name('16_generate_audited_figures.py')
s=importlib.util.spec_from_file_location('audited_figures',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.generate()
