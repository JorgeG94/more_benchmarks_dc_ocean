#!/usr/bin/env python3
"""dc_to_serialdo.py -- rewrite Fortran `do concurrent` into plain nested `do`.

The paper needs a *serial do-loop* baseline distinct from *serial do concurrent*:
same numerics, but the construct is ordinary `do` loops, so we can measure what
(if anything) the `do concurrent` construct itself costs a compiler, and so that
compilers without `do concurrent ... local(...)` support (e.g. gfortran <= 14)
can compile the code at all.

Transform, per loop:
    do concurrent(k=1:nz, j=1:ny, i=1:nx) local(a,b)    ->    do k=1,nz
       <body>                                                 do j=1,ny
    end do                                                    do i=1,nx
                                                                 <body>
                                                              end do
                                                              end do
                                                              end do

  * multi-index headers expand to nested loops (outer index written first);
  * the ONE `end do` that closed the concurrent loop becomes N `end do`s;
  * the `local(...)` clause is dropped -- in serial each iteration reuses the
    same storage, which is correct exactly when the loop was race-free (the
    invariant `do concurrent` already asserts).

This is a MECHANICAL transform; correctness is still gated downstream by the
bit-identity check (DC ref vs serial-do, max rel diff < 1e-12). The transformer
REFUSES (nonzero exit) on anything it cannot prove it handles -- a `do concurrent`
carrying a mask or a `reduce()` clause, or a `local_init(...)` -- rather than
silently mistransform. None exist in this repo today; the guard is a tripwire.
"""
import re
import sys

# a statement-opening DO that is NOT `do concurrent`: plain `do i=1,n`,
# `do while(...)`, bare `do`, or a construct-named `name: do ...`. Each needs
# exactly one `end do` to close.
_OPEN_DO = re.compile(r'^\s*(?:[A-Za-z]\w*\s*:\s*)?do\b(?!\s+concurrent\b)', re.I)
_OPEN_DC = re.compile(r'^\s*(?:[A-Za-z]\w*\s*:\s*)?do\s+concurrent\b', re.I)
_END_DO  = re.compile(r'^(\s*)end\s*do\b', re.I)
_COMMENT = re.compile(r'^\s*!')
# C-preprocessor conditionals. `#ifdef X / do concurrent(A) / #else /
# do concurrent(B) / #endif / <body> / end do` selects ONE header for a single
# shared body+`end do`. The transform sees both headers pre-cpp, so `#else`
# must RESET the loop-open stack to the `#if` snapshot -- the alternative branch
# re-opens the same construct, it is not an additional one.
_PP_IF    = re.compile(r'^\s*#\s*if')
_PP_ELSE  = re.compile(r'^\s*#\s*el(se|if)\b')
_PP_ENDIF = re.compile(r'^\s*#\s*endif\b')


def _split_top_level(s, sep=','):
    """Split `s` on `sep`, ignoring separators inside ()."""
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    out.append(cur)
    return out


def _index_specs(header):
    """From a full (continuation-joined) `do concurrent` statement, return the
    list of `do i=lo,hi[,step]` control lines. Raise on anything unhandled."""
    m = re.search(r'do\s+concurrent\s*\(', header, re.I)
    # find the balanced index-list parenthesis
    start = header.index('(', m.end() - 1)
    depth, end = 0, None
    for idx in range(start, len(header)):
        if header[idx] == '(':
            depth += 1
        elif header[idx] == ')':
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end is None:
        raise ValueError(f'unbalanced () in: {header!r}')
    inner = header[start + 1:end]
    trailer = header[end + 1:].strip()
    # trailer may only be a (dropped) local(...) clause -- reject the rest.
    low = trailer.lower()
    if low and not low.startswith('local('):
        raise ValueError(f'unhandled do-concurrent clause {trailer!r} in: {header!r}')
    if 'reduce(' in low or 'local_init(' in low:
        raise ValueError(f'unhandled locality clause in: {header!r}')

    controls = []
    for item in _split_top_level(inner):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:  # a mask expression -- not handled
            raise ValueError(f'do-concurrent mask/uncovered item {item!r} in: {header!r}')
        name, rng = item.split('=', 1)
        parts = _split_top_level(rng, ':')
        if len(parts) == 2:
            lo, hi = (p.strip() for p in parts)
            controls.append(f'do {name.strip()}={lo},{hi}')
        elif len(parts) == 3:
            lo, hi, st = (p.strip() for p in parts)
            controls.append(f'do {name.strip()}={lo},{hi},{st}')
        else:
            raise ValueError(f'bad index spec {item!r} in: {header!r}')
    return controls


def transform(text):
    lines = text.split('\n')
    out = []
    depth_stack = []      # closes-needed per currently-open loop construct
    pp_snap = []          # loop-stack snapshots at each open #if (for #else reset)
    prev_continued = False
    i = 0
    while i < len(lines):
        line = lines[i]

        # preprocessor conditionals gate loop-open bookkeeping (not the text).
        if not prev_continued and _PP_IF.match(line):
            pp_snap.append(list(depth_stack))
            out.append(line); i += 1; continue
        if not prev_continued and _PP_ELSE.match(line):
            if pp_snap:                      # re-enter: alternative re-opens same loops
                depth_stack = list(pp_snap[-1])
            out.append(line); i += 1; continue
        if not prev_continued and _PP_ENDIF.match(line):
            if pp_snap:
                pp_snap.pop()
            out.append(line); i += 1; continue

        # a continuation line, or a comment, is never itself a loop keyword
        if prev_continued or _COMMENT.match(line):
            out.append(line)
            prev_continued = line.rstrip().endswith('&') and not _COMMENT.match(line)
            i += 1
            continue

        if _OPEN_DC.match(line):
            indent = re.match(r'^(\s*)', line).group(1)
            # join continuation lines to see the whole header + local()
            joined, j = line, i
            while joined.rstrip().endswith('&'):
                joined = joined.rstrip()[:-1] + ' ' + lines[j + 1].strip()
                j += 1
            controls = _index_specs(joined)
            for c in controls:
                out.append(indent + c)
            depth_stack.append(len(controls))
            i = j + 1
            prev_continued = False
            continue

        if _OPEN_DO.match(line):
            depth_stack.append(1)
            out.append(line)
            prev_continued = line.rstrip().endswith('&')
            i += 1
            continue

        m = _END_DO.match(line)
        if m and depth_stack:
            n = depth_stack.pop()
            indent = m.group(1)
            out.append(line)                       # first close keeps any trailing name/comment
            for _ in range(n - 1):
                out.append(indent + 'end do')
            i += 1
            prev_continued = False
            continue

        out.append(line)
        prev_continued = line.rstrip().endswith('&') and not _COMMENT.match(line)
        i += 1

    if depth_stack:
        raise ValueError(f'unbalanced loops: {len(depth_stack)} unclosed at EOF')
    return '\n'.join(out)


def main(argv):
    if len(argv) != 3:
        sys.stderr.write('usage: dc_to_serialdo.py <in.F90> <out.F90>\n')
        return 2
    with open(argv[1]) as f:
        src = f.read()
    try:
        dst = transform(src)
    except ValueError as e:
        sys.stderr.write(f'dc_to_serialdo: {argv[1]}: {e}\n')
        return 1
    with open(argv[2], 'w') as f:
        f.write(dst)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
