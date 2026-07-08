import subprocess, os

codex = r'C:\Users\ASUS\AppData\Roaming\npm\codex.cmd'
result = subprocess.run(
    [codex, 'exec', 'echo hello world', '--ephemeral', '--skip-git-repo-check'],
    capture_output=True, text=True, timeout=60,
    cwd=r'D:\Desktop\evo\ChemEvoRAG_Phase1-main\exp'
)
print(f'stdout: {result.stdout[:800]}')
print(f'stderr: {result.stderr[:800]}')
print(f'rc: {result.returncode}')
