"""Unified Diff Patcher for Node 2 fast code repair.

Supports strict line-offset matching with fuzzy hunk fallback and syntax validation.
"""

import re

from hrm_agent.schemas import UnifiedDiffPatch


class Hunk:
    def __init__(
        self,
        old_start: int,
        old_len: int,
        new_start: int,
        new_len: int,
        lines: list[str],
    ):
        self.old_start = old_start
        self.old_len = old_len
        self.new_start = new_start
        self.new_len = new_len
        self.lines = lines  # Lines including prefix ' ', '+', '-'

    @property
    def old_slice(self) -> list[str]:
        """Original lines expected in source code (prefix ' ' or '-')."""
        return [line[1:] for line in self.lines if line.startswith(" ") or line.startswith("-")]

    @property
    def new_slice(self) -> list[str]:
        """Replacement lines to produce in target code (prefix ' ' or '+')."""
        return [line[1:] for line in self.lines if line.startswith(" ") or line.startswith("+")]


def parse_unified_diff(diff_text: str) -> tuple[str, str, list[Hunk]]:
    """Parse a unified diff into source/target filenames and list of Hunks."""
    source_file = "solution.cpp"
    target_file = "solution.cpp"
    hunks: list[Hunk] = []

    lines = diff_text.strip().splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("--- a/"):
            source_file = line[6:].strip()
            idx += 1
            continue
        elif line.startswith("--- "):
            source_file = line[4:].strip()
            idx += 1
            continue
        elif line.startswith("+++ b/"):
            target_file = line[6:].strip()
            idx += 1
            continue
        elif line.startswith("+++ "):
            target_file = line[4:].strip()
            idx += 1
            continue

        hunk_header_match = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
        if hunk_header_match:
            old_start = int(hunk_header_match.group(1))
            old_len = int(hunk_header_match.group(2)) if hunk_header_match.group(2) is not None else 1
            new_start = int(hunk_header_match.group(3))
            new_len = int(hunk_header_match.group(4)) if hunk_header_match.group(4) is not None else 1

            idx += 1
            hunk_lines: list[str] = []
            while idx < len(lines):
                cur = lines[idx]
                if cur.startswith("@@") or cur.startswith("--- ") or cur.startswith("diff "):
                    break
                if cur and cur[0] in (" ", "+", "-", "\\"):
                    hunk_lines.append(cur)
                    idx += 1
                elif cur == " ":
                    hunk_lines.append(" ")
                    idx += 1
                else:
                    break
            hunks.append(Hunk(old_start, old_len, new_start, new_len, hunk_lines))
        else:
            idx += 1

    return source_file, target_file, hunks


def validate_syntax_balance(code: str, language: str = "cpp") -> tuple[bool, str | None]:
    """Basic balance check for braces and brackets to avoid catastrophic diffs."""
    stack = []
    pairs = {"{": "}", "(": ")", "[": "]"}
    in_multi_line_comment = False
    in_string = False
    string_char = ""

    lines = code.split("\n")
    for row, line in enumerate(lines, 1):
        i = 0
        in_string = False
        string_char = ""
        while i < len(line):
            ch = line[i]
            if not in_multi_line_comment and not in_string:
                if language == "cpp" and line[i : i + 2] == "//":
                    break
                if language == "py" and ch == "#":
                    break
                if language == "cpp" and line[i : i + 2] == "/*":
                    in_multi_line_comment = True
                    i += 2
                    continue
                if ch in ('"', "'"):
                    in_string = True
                    string_char = ch
                    i += 1
                    continue
                if ch in pairs:
                    stack.append((ch, row))
                elif ch in pairs.values():
                    if not stack:
                        return False, f"Unmatched closing '{ch}' at line {row}"
                    top, start_row = stack.pop()
                    if pairs[top] != ch:
                        return (
                            False,
                            f"Mismatched '{top}' (line {start_row}) with '{ch}' at line {row}",
                        )
            elif in_multi_line_comment:
                if line[i : i + 2] == "*/":
                    in_multi_line_comment = False
                    i += 2
                    continue
            elif in_string:
                if ch == "\\" and i + 1 < len(line):
                    i += 2
                    continue
                if ch == string_char:
                    in_string = False
            i += 1

    if stack:
        top, row = stack[-1]
        return False, f"Unclosed '{top}' opened at line {row}"

    return True, None


def apply_patch(original_code: str, raw_diff: str, language: str = "cpp") -> tuple[bool, str, UnifiedDiffPatch]:
    """
    Apply a unified diff to original_code with strict offset and fuzzy fallback.

    Returns:
        (success, patched_code, patch_metadata)
    """
    src_file, tgt_file, hunks = parse_unified_diff(raw_diff)
    patch_meta = UnifiedDiffPatch(
        raw_diff=raw_diff,
        source_file=src_file,
        target_file=tgt_file,
        hunks_count=len(hunks),
        applied_cleanly=False,
    )

    if not hunks:
        patch_meta.validation_error = "No valid diff hunks parsed from text"
        return False, original_code, patch_meta

    src_lines = original_code.split("\n")
    modified_lines = list(src_lines)
    all_clean = True

    # Sort hunks in descending order of old_start so offsets earlier in the file don't shift later hunks
    sorted_hunks = sorted(hunks, key=lambda h: h.old_start, reverse=True)

    for hunk in sorted_hunks:
        old_expected = hunk.old_slice
        new_replacement = hunk.new_slice

        # 1. Strict match at 1-based old_start
        hunk_applied = False
        target_idx = max(0, hunk.old_start - 1)

        # Check if strict match aligns
        if target_idx + len(old_expected) <= len(modified_lines):
            cand = modified_lines[target_idx : target_idx + len(old_expected)]
            if cand == old_expected:
                modified_lines[target_idx : target_idx + len(old_expected)] = new_replacement
                hunk_applied = True

        # 2. Fuzzy match within local neighborhood (+/- 30 lines)
        if not hunk_applied and old_expected:
            all_clean = False
            best_idx = -1
            best_score = 0.0

            search_start = max(0, target_idx - 30)
            search_end = min(len(modified_lines) - len(old_expected) + 1, target_idx + 30)

            for idx in range(search_start, search_end):
                cand = modified_lines[idx : idx + len(old_expected)]
                match_count = sum(1 for c, e in zip(cand, old_expected) if c == e)
                norm_match = sum(1 for c, e in zip(cand, old_expected) if c.strip() == e.strip())
                score = (match_count * 2.0 + norm_match) / (3.0 * len(old_expected))
                if score > best_score:
                    best_score = score
                    best_idx = idx

            # If strong fuzzy match found (>= 75% line match)
            if best_idx >= 0 and best_score >= 0.75:
                modified_lines[best_idx : best_idx + len(old_expected)] = new_replacement
                hunk_applied = True

        # 3. Global search if still not applied
        if not hunk_applied and old_expected:
            for idx in range(len(modified_lines) - len(old_expected) + 1):
                cand = modified_lines[idx : idx + len(old_expected)]
                if [c.strip() for c in cand] == [e.strip() for e in old_expected]:
                    modified_lines[idx : idx + len(old_expected)] = new_replacement
                    hunk_applied = True
                    break

        if not hunk_applied:
            patch_meta.validation_error = f"Failed to match hunk @@ -{hunk.old_start},{hunk.old_len} in target code"
            return False, original_code, patch_meta

    result_code = "\n".join(modified_lines)

    # 4. Syntax balance validation
    syntax_ok, syntax_err = validate_syntax_balance(result_code, language)
    if not syntax_ok:
        patch_meta.validation_error = f"Syntax validation failed after patch: {syntax_err}"
        return False, original_code, patch_meta

    patch_meta.applied_cleanly = all_clean
    return True, result_code, patch_meta
