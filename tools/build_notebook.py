"""Assemble the submission notebook from the documented sources.

Sources (cells separated by `# %%`, markdown cells marked `# %% [markdown]`):
  dicke_solution.py          -> solution (replaces the create_dicke_state stub) and bonus (replaces the adaptive stub)
  tools/nb_verification.py   -> verification cells (after the fake-calibration cells)
  tools/nb_emulator.py       -> emulator + design experiments
  tools/nb_bonus_scoring.py  -> local validate_circuit + bonus scoring run
  tools/nb_disclosure.py     -> AI-usage disclosure
The organisers' cells come from tools/challenge_template.ipynb, so the build is repeatable.
Run from the repository root:  .venv/bin/python tools/build_notebook.py
"""
import json
import re
import uuid

NB = '[standard_track] Google Quantum AI Challenge.ipynb'
TEMPLATE = 'tools/challenge_template.ipynb'  # pristine copy of the organisers' notebook


def parse(path):
    """Split a `# %%`-annotated source file into (cell_type, source) pairs."""
    text = open(path).read()
    chunks = re.split(r'^# %%(.*)$', text, flags=re.M)
    cells = []
    for i in range(1, len(chunks), 2):
        header, body = chunks[i], chunks[i + 1].strip('\n')
        if '[markdown]' in header:
            lines = [ln[2:] if ln.startswith('# ') else ln.lstrip('#') for ln in body.split('\n')]
            cells.append(('markdown', '\n'.join(lines).strip('\n')))
        elif body.strip():
            cells.append(('code', body))
    return cells


def make(cell_type, source):
    cell = {'cell_type': cell_type, 'id': uuid.uuid4().hex[:8], 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if cell_type == 'code':
        cell.update({'execution_count': None, 'outputs': []})
    return cell


def main():
    nb = json.load(open(TEMPLATE))
    orig = nb['cells']
    src = lambda c: ''.join(c['source'])
    assert src(orig[2]).startswith('def create_dicke_state(') and src(orig[14]).startswith('def create_dicke_state_adaptive(')
    assert src(orig[16]).startswith('def score_solution()')
    solution = parse('dicke_solution.py')
    cut = next(i for i, (t, s) in enumerate(solution) if t == 'markdown' and s.startswith('## 6. Bonus'))
    main_cells, bonus_cells = solution[:cut], solution[cut:]
    fixed4 = src(orig[4]).replace('    keep_indices = range(qubits)\n',
                                  '    keep_indices = range(len(qubits))  # FIX (DuQuantum): range(qubits) raised TypeError on a set\n')
    assert fixed4 != src(orig[4]), 'test_circuit bug not found'
    scoring = parse('tools/nb_bonus_scoring.py')
    keep = lambda i: dict(orig[i], outputs=[], execution_count=None) if orig[i]['cell_type'] == 'code' else orig[i]
    cells = [keep(0), keep(1)]
    cells += [make(t, s) for t, s in main_cells]
    cells += [keep(3), make('code', fixed4)]
    cells += [keep(i) for i in range(5, 13)]
    cells += [make(t, s) for t, s in parse('tools/nb_verification.py')]
    cells += [make(t, s) for t, s in parse('tools/nb_emulator.py')]
    cells += [keep(13)]
    cells += [make(t, s) for t, s in bonus_cells]
    cells += [keep(15)] + [make(t, s) for t, s in scoring[:2]] + [keep(16)] + [make(t, s) for t, s in scoring[2:]]
    cells += [keep(17)]
    cells += [make(t, s) for t, s in parse('tools/nb_disclosure.py')]
    for c in cells:
        c.setdefault('id', uuid.uuid4().hex[:8])
    nb['cells'] = cells
    nb['nbformat'], nb['nbformat_minor'] = 4, 5  # cell ids require nbformat 4.5
    json.dump(nb, open(NB, 'w'), indent=1, ensure_ascii=False)
    open(NB, 'a').write('\n')
    print(f'wrote {len(cells)} cells')


if __name__ == '__main__':
    main()
