"""README.md의 '## 판단' ~ '## 파일' 앞 부분을 노트북 마지막 결론 셀에 넣는다. 사용: python insert_conclusion.py <폴더> <노트북 파일명>"""
import sys
from pathlib import Path
import nbformat as nbf
d, name = Path(sys.argv[1]), sys.argv[2]
r = open(d / "README.md", encoding="utf-8").read()
c = r[r.index("## 판단"):r.index("## 파일")].replace("\n## ", "\n### ").replace("## 판단", "### 판단")
nb = nbf.read(d / name, 4)
assert nb.cells[-1].cell_type == "markdown"
nb.cells[-1].source = "결과 표와 그래프는 위 절 참고.\n\n" + c
nbf.write(nb, d / name)
b = d / "build_nb.py"; s = open(b, encoding="utf-8").read()
s = s.replace('md("(실행 결과를 보고 작성 — README.md와 같은 내용)")',
              'r_ = open("README.md", encoding="utf-8").read()\nmd(r_[r_.index("## 판단"):r_.index("## 파일")].replace("\\n## ", "\\n### ").replace("## 판단", "### 판단"))')
open(b, "w", encoding="utf-8").write(s); print("ok")
