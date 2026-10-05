import json

import pytest

from token_prediction.collection.codex_runner import _submission_marker_present
from token_prediction.collection.dsh.external_task_pack import EXTERNAL_TASK_SUBMISSION_MARKER


def event(method, **params):
    return {"message": {"method": method, "params": {
        "threadId": "thread", "turnId": "turn", **params,
    }}}


@pytest.mark.parametrize("case,expected", [
    ("completed", True), ("open_pty", True), ("command_echo", False),
    ("assistant_text", False), ("other_turn", False), ("separate_commands", False),
])
def test_marker_uses_output_from_same_command_and_turn(tmp_path, case, expected):
    marker = EXTERNAL_TASK_SUBMISSION_MARKER
    patch = "diff --git a/source.py b/source.py\n"
    events = [event("item/completed", item={
        "id": "cmd", "type": "commandExecution", "status": "completed",
        "aggregatedOutput": f"{marker}\n{patch}",
    })]
    if case == "open_pty":
        # ANSI codes and even the marker may span notification chunks. The
        # interactive shell has no command-completed event before turn completion.
        events = [event("item/commandExecution/outputDelta", itemId="cmd", delta=part)
                  for part in ["warning\r\n\x1b[38;5;9m", marker[:12],
                               marker[12:] + "\r\x1b[m", "\n\x1b[?25l" + patch]]
    elif case == "command_echo":
        events[0]["message"]["params"]["item"]["aggregatedOutput"] = (
            f"PS> Write-Output '{marker}'; Write-Output $patch\n{patch}"
        )
    elif case == "assistant_text":
        events[0]["message"]["params"]["item"]["type"] = "agentMessage"
    elif case == "other_turn":
        events[0]["message"]["params"]["turnId"] = "other"
    elif case == "separate_commands":
        events = [event("item/commandExecution/outputDelta", itemId=cmd, delta=part)
                  for cmd, part in [("one", marker + "\n"), ("two", patch)]]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    assert _submission_marker_present(path, "thread", "turn") is expected
