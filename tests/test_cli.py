import json
from pathlib import Path
import subprocess
import sys


def test_cli_demo_outputs_and_resume(tmp_path):
    output=tmp_path/'run'
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','mycelia','--steps','15','--size','24','--output',str(output)],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (output/'report.html').exists()
    assert '<svg' in (output/'network.svg').read_text()
    summary=json.loads((output/'summary.json').read_text())
    assert summary['step']==15
    result=subprocess.run([sys.executable,'main.py','--steps','5','--resume',str(output/'state.json'),'--output',str(output)],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads((output/'summary.json').read_text())['step']==20
