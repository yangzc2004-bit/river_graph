# 用法: python submit.py slide01.xml
# 先跑 xml_lint，error_count=0 才提交到飞书幻灯片
import subprocess, sys, json, pathlib

SKILL = r'C:\Users\17493\AppData\Local\Doubao\User Data\Default\.doubao\agent_mode\workspace\.skills\ppt'
LINT = pathlib.Path(SKILL) / 'scripts' / 'xml_lint.py'
PID = 'EiRmsud2NlHZW9dUhrJcNLYqnhh'

f = sys.argv[1]
r = subprocess.run(['python', str(LINT), '--input', f], capture_output=True, text=True, encoding='utf-8')
out = r.stdout
print(out)
try:
    # 找到 JSON 起始位置
    start = out.find('{')
    report = json.loads(out[start:])
    errs = report.get('summary', {}).get('error_count', 0)
except Exception as e:
    print('LINT PARSE FAIL:', e)
    sys.exit(2)

if errs != 0:
    print(f'== {f} has {errs} errors, NOT submitted ==')
    sys.exit(1)

print(f'== {f} lint clean, adding slide ==')
a = subprocess.run(['lark-cli', 'slides', '+add-slide', '--presentation', PID, '--slide', '@' + f],
                   capture_output=True, text=True, encoding='utf-8')
print(a.stdout)
if a.stderr:
    print('STDERR:', a.stderr[:800])
