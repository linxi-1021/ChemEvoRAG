"""Codex-style Agent baseline — LLM as autonomous agent with grep/read/bash tools over mineru markdown.

The agent sees the full directory of parsed papers (MinerU .md files), and uses tools to:
  - list_dir / glob: find files
  - grep: search across all papers for keywords/SMILES/names
  - read: read specific sections of a paper
  - answer: submit final answer with evidence citations

This replicates the Claude Code / Codex agent pattern: an LLM loop with tool access,
limited to 5 rounds.

Run: C:/Users/ASUS/.conda/envs/rag_direct/python.exe run_agent_codex.py --limit 118
"""
import io,os,sys,time,certifi,json,re,fnmatch
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.environ['SSL_CERT_FILE']=certifi.where()

K='sk-gr-02f44cd3404411cd11cdd88da080685ac1c54986'; B='https://endpoint.greatrouter.com'; M='DeepSeek-V3'
os.environ['OPENAI_API_KEY']=K; os.environ['API_KEY']=K; os.environ['BASE_URL']=B; os.environ['LLM_MODEL']=M

from eval_utils import load_all_questions, save_results, PDF_DIR
from pathlib import Path
from openai import OpenAI
import argparse

p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=118)
p.add_argument('--max-rounds',type=int,default=5)
a=p.parse_args()
qs=load_all_questions()
if a.limit<118: qs=qs[:a.limit]
print(f'Codex Agent: {len(qs)} questions, max {a.max_rounds} rounds')

client=OpenAI(api_key=K,base_url=B)

# Build the paper directory index
MINERU_DIR = Path('D:/Desktop/evo/ChemEvoRAG_Phase1-main/data/mineru_output')
paper_dirs = sorted([d for d in MINERU_DIR.iterdir() if d.is_dir() and (d / 'extracted' / 'full.md').exists()])
PAPER_INDEX = {}
for pd in paper_dirs:
    md_path = pd / 'extracted' / 'full.md'
    with open(md_path, encoding='utf-8') as f:
        content = f.read()
    name = pd.name
    PAPER_INDEX[name] = {'path': str(md_path), 'content': content, 'title': '', 'size': len(content)}
    # Extract title from first # heading
    m = re.search(r'^# (.+)$', content, re.MULTILINE)
    if m: PAPER_INDEX[name]['title'] = m.group(1).strip()

print(f'  {len(PAPER_INDEX)} papers indexed ({sum(v["size"] for v in PAPER_INDEX.values()):,} chars total)')

# ── Tool implementations ──
def tool_list_papers(query: str = '') -> str:
    """List available papers. Optional query filters by title."""
    lines = []
    for pid, info in sorted(PAPER_INDEX.items()):
        title = info['title']
        if not query or query.lower() in title.lower() or query.lower() in pid.lower():
            lines.append(f"  [{pid}] {title[:100]} ({info['size']:,} chars)")
    return '\n'.join(lines[:30]) if lines else '(no matches)'

def tool_grep(pattern: str, paper_id: str = '') -> str:
    """Search for pattern (regex or literal) in all papers or specific paper. Returns matching lines with context."""
    results = []
    papers = [paper_id] if paper_id and paper_id in PAPER_INDEX else sorted(PAPER_INDEX.keys())
    for pid in papers:
        content = PAPER_INDEX[pid]['content']
        # Try regex first, fall back to literal
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except:
            compiled = re.compile(re.escape(pattern), re.IGNORECASE)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if compiled.search(line):
                # Get context: 2 lines before and after
                start = max(0, i-2)
                end = min(len(lines), i+3)
                ctx = ' | '.join(lines[start:end])
                results.append(f"[{pid}:L{i+1}] {ctx[:300]}")
                if len(results) >= 30:
                    break
        if len(results) >= 30:
            break
    if not results:
        return f'(no matches for pattern: {pattern})'
    return '\n'.join(results[:30])

def tool_read(paper_id: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read a section of a paper. If start=0, read from beginning. If end=0, read to end."""
    if paper_id not in PAPER_INDEX:
        return f'(paper {paper_id} not found)'
    content = PAPER_INDEX[paper_id]['content']
    lines = content.split('\n')
    s = max(0, start_line)
    e = min(len(lines), end_line) if end_line > 0 else len(lines)
    if s == 0 and e == len(lines):
        # Return structure overview: headings + tables only
        overview = []
        for i, line in enumerate(lines):
            if line.startswith('#') or '<table>' in line.lower() or line.strip().startswith('Table'):
                overview.append(f"  L{i+1}: {line[:150]}")
        return f"STRUCTURE of [{paper_id}] ({len(lines)} lines):\n" + '\n'.join(overview[:50])
    return '\n'.join(f"L{i+1}: {line}" for i, line in enumerate(lines[s:e], s))

def tool_read_section(paper_id: str, section_name: str) -> str:
    """Read a specific section (by heading name) from a paper."""
    if paper_id not in PAPER_INDEX:
        return f'(paper {paper_id} not found)'
    content = PAPER_INDEX[paper_id]['content']
    lines = content.split('\n')
    # Find section
    in_section = False
    section_lines = []
    for line in lines:
        if line.startswith('#'):
            if section_name.lower() in line.lower():
                in_section = True
                section_lines.append(line)
                continue
            elif in_section:
                break
        if in_section:
            section_lines.append(line)
    if not section_lines:
        return f'(section "{section_name}" not found in {paper_id})'
    return '\n'.join(section_lines[:100])

def tool_read_table(paper_id: str, table_num: str) -> str:
    """Read a specific table from a paper. table_num like '1' or '2'."""
    if paper_id not in PAPER_INDEX:
        return f'(paper {paper_id} not found)'
    content = PAPER_INDEX[paper_id]['content']
    lines = content.split('\n')
    # Find table marker
    target = f'Table {table_num}.'
    in_table = False
    table_lines = []
    for line in lines:
        if target in line and not in_table:
            in_table = True
            table_lines.append(line)
            continue
        if in_table:
            if line.strip().startswith('<table>'):
                table_lines.append(line)
            elif line.strip().startswith('</table>') or line.strip().startswith('</tr></table>'):
                table_lines.append(line)
                break
            elif line.strip().startswith('#'):
                break
            else:
                table_lines.append(line)
    if not table_lines:
        return f'(Table {table_num} not found in {paper_id}. Try tool_read for paper structure.)'
    return '\n'.join(table_lines[:80])

TOOLS_DESC = """You have these tools (respond with a JSON tool call: {"tool": "name", "args": {...}}):

1. **list_papers** — List all papers.  args: {"query": "optional filter"}
2. **grep** — Search across papers.  args: {"pattern": "regex or text", "paper_id": "optional specific paper"}
3. **read** — Read paper structure (headings+table list).  args: {"paper_id": "X"}
4. **read_section** — Read one section.  args: {"paper_id": "X", "section_name": "Results"}
5. **read_table** — Read a specific table.  args: {"paper_id": "X", "table_num": "1"}
6. **answer** — Submit final answer.  args: {"text": "your answer with evidence citations", "confidence": 0.8}

STRATEGY GUIDE:
- For "what does label X refer to?" questions: first grep for "X" to find the paper, then grep for the label in that paper, then read context around the match
- For "which solvent gave higher yield?" questions: read the relevant table, compare the numbers, cite the table
- For entity lookups: grep for the compound name or SMILES
- Use 2-3 tool calls max per question
- If you can't find the answer after 2-3 searches, submit answer with what you found and confidence=0.1

Reply ONLY with a JSON tool call per round. The system will execute it and return the result."""

def execute_tool(tool_name: str, args: dict) -> str:
    if tool_name == 'list_papers':
        return tool_list_papers(args.get('query', ''))
    elif tool_name == 'grep':
        return tool_grep(args.get('pattern', ''), args.get('paper_id', ''))
    elif tool_name == 'read':
        return tool_read(args.get('paper_id', ''), args.get('start_line', 0), args.get('end_line', 0))
    elif tool_name == 'read_section':
        return tool_read_section(args.get('paper_id', ''), args.get('section_name', ''))
    elif tool_name == 'read_table':
        return tool_read_table(args.get('paper_id', ''), args.get('table_num', ''))
    else:
        return f'Unknown tool: {tool_name}'

# ── Agent loop ──
SYSTEM = f"""You are a chemistry research agent. You have access to {len(PAPER_INDEX)} parsed chemistry papers in markdown format.
{TOOLS_DESC}
"""

def agent_answer(question: str) -> tuple[str, float, str]:
    """Run agent loop and return (answer, confidence, trace_log)."""
    messages = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'QUESTION: {question}\n\nStart by listing papers, then grep/search for relevant information. Use at most {a.max_rounds} rounds. Respond with JSON tool calls only.'}]
    trace = []

    for round_num in range(a.max_rounds):
        try:
            resp = client.chat.completions.create(
                model=M, messages=messages, temperature=0.0, max_tokens=2048)
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            trace.append(f'Round {round_num+1}: API error {e}')
            break

        # Parse JSON from response
        try:
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'): raw = raw[4:]
                raw = raw.strip()
            tool_call = json.loads(raw)
        except json.JSONDecodeError:
            trace.append(f'Round {round_num+1}: bad JSON — {raw[:200]}')
            # Try to extract answer if it looks like one
            if 'answer' in raw.lower():
                return raw, 0.1, '\n'.join(trace)
            continue

        tool_name = tool_call.get('tool', '')
        args = tool_call.get('args', {})

        if tool_name == 'answer':
            ans_text = args.get('text', '') or args.get('answer', '')
            conf = float(args.get('confidence', 0.5))
            trace.append(f'Round {round_num+1}: ANSWER (conf={conf})')
            return ans_text, conf, '\n'.join(trace)

        # Execute tool
        result = execute_tool(tool_name, args)
        trace.append(f'Round {round_num+1}: {tool_name}({json.dumps(args)}) -> {len(result)} chars')
        # Truncate long results
        if len(result) > 3000:
            result = result[:3000] + f'\n... (truncated, {len(result)} chars total)'
        messages.append({'role': 'assistant', 'content': raw})
        messages.append({'role': 'user', 'content': f'TOOL RESULT:\n{result}\n\nContinue with next tool call or submit answer.'})

    # Max rounds exceeded — try to extract answer from last message
    try:
        resp = client.chat.completions.create(
            model=M, messages=messages + [{'role': 'user', 'content': 'Max rounds reached. Provide your best answer now as JSON: {"tool":"answer","args":{"text":"...","confidence":0.X}}'}],
            temperature=0.0, max_tokens=1024)
        raw = resp.choices[0].message.content.strip()
        if raw.startswith('```'): raw = raw.split('```')[1].split('```')[0].strip()
        tool_call = json.loads(raw)
        return tool_call.get('args',{}).get('text',''), float(tool_call.get('args',{}).get('confidence',0.1)), '\n'.join(trace)
    except:
        return '(agent failed to produce answer)', 0.0, '\n'.join(trace)

# ── Run ──
rs=[]
for i,q in enumerate(qs):
    print(f'  [{i+1}/{len(qs)}] {q["id"]}  intent={q["intent"]}',flush=True)
    t0=time.time()
    try:
        ans,conf,trace=agent_answer(q['question'])
    except Exception as e:
        ans=f'[ERROR] {e}'; conf=0.0; trace=''
    elapsed=round(time.time()-t0,2)
    print(f'    -> rounds={trace.count("Round")}, conf={conf:.2f}, elapsed={elapsed}s')
    rs.append({'question_id':q['id'],'paper_id':q['paper_id'],'paper_title':q['paper_title'],
        'intent':q['intent'],'question':q['question'],'ground_truth':q['ground_truth_answer'],
        'key_entities':q.get('key_entities',[]),'system_answer':str(ans).strip(),
        'elapsed_sec':elapsed})

sys.path.insert(0,'D:/Desktop/evo/ChemEvoRAG_Phase1-main/src')
from judge import judge_answer
print('Scoring...')
for i,r in enumerate(rs):
    j=judge_answer(r['system_answer'],r['ground_truth'],r.get('key_entities',[]))
    r['judge_score']=j.get('score',0); r['entities_found']=j.get('key_entities_found',[]); r['entities_missing']=j.get('key_entities_missing',[]); r['judge_reasoning']=j.get('reasoning','')
    if (i+1)%25==0: print(f'  {i+1}/{len(rs)}')
s=[r['judge_score'] for r in rs]; print(f'\nCodex Agent: {sum(s)/len(s):.3f}')
save_results('agent_codex',rs,{'model':M,'max_rounds':a.max_rounds})
print('Done')
