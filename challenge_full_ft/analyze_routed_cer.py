#!/usr/bin/env python3
"""在被路由的样本子集上，对比 baseline 与 specialist 的字级 CER。
仅用于诊断"专用模型在被选中的样本上到底比 baseline 好多少"。
"""
import json, re, sys
from collections import defaultdict

def norm(s):
    if s is None: return ''
    s = str(s)
    s = re.sub(r'<\|[^|]*\|>', '', s)
    s = re.sub(r'\[[^\]]*\]', '', s)
    s = re.sub(r'[\s\u3000]+', '', s)
    s = re.sub(r'[，。！？、；：,.!?;:"\'`~\-—…·]+', '', s)
    return s.lower()

def cer_err(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    m, n = len(ref), len(hyp)
    if m == 0:
        return (n, 0)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = prev if ref[i-1] == hyp[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return (dp[n], m)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

BASE_PRED = 'outputs_vc_v2/pred_test_conf.jsonl'
base_map = {r['utt_id']: r for r in load_jsonl(BASE_PRED)}

for tag in ('p95', 'p99'):
    routed = load_jsonl(f'outputs_hybrid_specialist/{tag}/pred_test_routed.jsonl')
    agg = defaultdict(lambda: {'n': 0, 'eb': 0, 'lb': 0, 'eh': 0, 'lh': 0,
                                'win': 0, 'tie': 0, 'lose': 0})
    for r in routed:
        route = r.get('route_used', 'vc_v2')
        if route == 'vc_v2':
            continue
        uid = r['utt_id']
        ref = r.get('ref_text', '')
        hb = base_map[uid].get('pred_text', '')
        hh = r.get('pred_text', '')
        eb, lb = cer_err(ref, hb)
        eh, lh = cer_err(ref, hh)
        a = agg[route]
        a['n'] += 1
        a['eb'] += eb; a['lb'] += lb
        a['eh'] += eh; a['lh'] += lh
        if eh < eb: a['win'] += 1
        elif eh > eb: a['lose'] += 1
        else: a['tie'] += 1

    print(f"\n===== {tag} =====")
    print(f"  {'route':<24}{'n':>5}  {'CER_base':>10}  {'CER_hyb':>10}  {'delta':>10}  win/tie/lose")
    total = {'n': 0, 'eb': 0, 'lb': 0, 'eh': 0, 'lh': 0, 'win': 0, 'tie': 0, 'lose': 0}
    for k in sorted(agg):
        v = agg[k]
        cb = v['eb'] / v['lb'] * 100 if v['lb'] else 0
        ch = v['eh'] / v['lh'] * 100 if v['lh'] else 0
        print(f"  {k:<24}{v['n']:>5}  {cb:>9.2f}%  {ch:>9.2f}%  {ch-cb:+9.2f}pp  {v['win']}/{v['tie']}/{v['lose']}")
        for kk in total:
            total[kk] += v[kk]
    if total['n']:
        cb = total['eb'] / total['lb'] * 100
        ch = total['eh'] / total['lh'] * 100
        print(f"  {'ALL_ROUTED':<24}{total['n']:>5}  {cb:>9.2f}%  {ch:>9.2f}%  {ch-cb:+9.2f}pp  {total['win']}/{total['tie']}/{total['lose']}")

# 详细拆解 p95 的 win/lose 样本，看看具体是什么类型的失败
print("\n===== p95 详细 win/lose 案例（各 route 取前 3 条 lose）=====")
routed = load_jsonl('outputs_hybrid_specialist/p95/pred_test_routed.jsonl')
lose_cases = defaultdict(list)
for r in routed:
    route = r.get('route_used', 'vc_v2')
    if route == 'vc_v2': continue
    uid = r['utt_id']; ref = r.get('ref_text', '')
    hb = base_map[uid].get('pred_text', ''); hh = r.get('pred_text', '')
    eb, _ = cer_err(ref, hb); eh, _ = cer_err(ref, hh)
    if eh > eb:
        lose_cases[route].append((eh - eb, uid, ref, hb, hh))
for k in sorted(lose_cases):
    lose_cases[k].sort(reverse=True)
    print(f"\n[{k}] lose top-3 by delta_err:")
    for delta, uid, ref, hb, hh in lose_cases[k][:3]:
        print(f"  utt_id={uid}  +{delta} errs")
        print(f"    REF : {ref[:120]}")
        print(f"    BASE: {hb[:120]}")
        print(f"    HYB : {hh[:120]}")
