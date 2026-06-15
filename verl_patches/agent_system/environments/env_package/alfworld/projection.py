# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import re

# Robust action-tag matchers. The original parser ONLY accepted the strict
# `<action>...</action>`. Weak / not-yet-format-trained models (vanilla base,
# k=1 latents) frequently emit the SAME action content with a near-miss wrapper
# \u2014 `[action]...[/action]`, `[action>...</action>]`, mixed brackets \u2014 and the
# strict parser rejected every one of them, deflating those models to ~0% while
# the RL ckpt (always `<action>`) scored fine. Tag style is a formatting artifact,
# not task capability, so we extract the content from any `<`/`[` action wrapper.
# Strict `<action>...</action>` remains a subset, so format-clean models (RL ckpt)
# are unaffected (validated: RL ckpt stays 87.9%).
_ACTION_PAIR = re.compile(r'[<\[]\s*action\s*[>\]](.*?)[<\[]\s*/\s*action\s*[>\]]', re.DOTALL)
_ACTION_OPEN = re.compile(r'[<\[]\s*action\s*[>\]](.*)', re.DOTALL)
_THINK_OPEN = re.compile(r'[<\[]\s*think\s*[>\]]')


def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        low = actions[i].lower()

        # Extract action content from any <action>/[action] style wrapper.
        m = _ACTION_PAIR.search(low)
        if m is not None:
            actions[i] = m.group(1).strip().lower()
            valids[i] = 1
        else:
            # Open tag present but malformed/missing close: take the remainder
            # of the line after the open tag (handles `<action>go to x` truncations).
            m2 = _ACTION_OPEN.search(low)
            if m2 is not None:
                cand = m2.group(1).strip().lower()
                cand = re.split(r'[<\[]', cand)[0].strip()  # cut any trailing junk tag
                if cand:
                    actions[i] = cand
                    valids[i] = 1
                else:
                    actions[i] = low[-30:]
            else:
                actions[i] = low[-30:]  # no action wrapper at all -> invalid

        # check think block (accept <think> or [think])
        if _THINK_OPEN.search(original_str) is None:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids
