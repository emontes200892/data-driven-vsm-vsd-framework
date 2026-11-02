"""Export curated audited results/discussion; do not invent narrative from scores."""
from pathlib import Path
r=Path(__file__).resolve().parents[1];p=r.parent/'manuscript/main.tex'
s=p.read_text();a=s.index('\\section{Results}');b=s.index('\\section{Conclusions and Future Work}')
(r/'results/results_discussion_verified.tex').write_text(s[a:b])
