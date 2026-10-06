"""LLM natural-language parsing constrained to a small skill-plan schema."""

import json
import os
import re
from urllib.error import URLError
from urllib.request import Request, urlopen


OBJECTS = ('red_cube', 'yellow_cube', 'blue_cube', 'green_cube', 'purple_cube')
ZONES = ('zone_a', 'zone_b', 'zone_c')


class LLMPlanner:
    """Ask an OpenAI-compatible endpoint for pick/place/home skills.

    The response is treated as untrusted structured data. It cannot contain
    coordinates or joint values, and is checked against the user's request
    schema before environment-aware planning continues.
    """

    def __init__(self):
        self.base_url = os.environ.get('LLM_BASE_URL', 'http://localhost:20128/v1').rstrip('/')
        self.model = os.environ.get('LLM_MODEL', 'oc/muse-spark-1.3-contributor-free')
        self.api_key = os.environ.get('LLM_API_KEY', 'none')
        self.enabled = os.environ.get('LLM_ENABLED', 'true').lower() not in ('0', 'false', 'no')

    def generate_core_plan(self, command, state):
        """Return a validated direct plan and its source label."""
        if self.enabled:
            try:
                plan = self._request_llm(command, state)
                self.validate_core_plan(plan)
                locally_understood = self._parse_locally(command)
                if locally_understood is not None and plan != locally_understood:
                    return locally_understood, 'local-parser (LLM plan did not match explicit request)'
                return plan, 'llm'
            except Exception as exc:
                fallback = self._parse_locally(command)
                if fallback is None:
                    raise ValueError(
                        'LLM không trả plan hợp lệ và local parser không hiểu lệnh: '
                        f'{exc}') from exc
                return fallback, f'local-parser ({exc})'

        fallback = self._parse_locally(command)
        if fallback is None:
            raise ValueError('LLM_ENABLED=false và local parser không hiểu lệnh.')
        return fallback, 'local-parser (LLM disabled)'

    def _request_llm(self, command, state):
        system = (
            'You are the language-to-skill planner for a UR3 tabletop robot. '
            'Return only JSON. Choose exactly one visible object and one valid zone. '
            'Do not return coordinates, poses, joint values, trajectories, shell commands, '
            'or any other fields. The only skills are pick, place, home, in that order. '
            'Use the current camera observations to avoid referring to an absent object. '
            'The runtime will independently check occupancy and may add safe clearing skills.'
        )
        user = {
            'request': command,
            'visible_objects': sorted(state.get('objects', {}).keys()),
            'zones': state.get('zones', {}),
            'allowed_objects': list(OBJECTS),
            'allowed_zones': list(ZONES),
            'required_json_shape': {
                'plan': [
                    {'skill': 'pick', 'object': 'red_cube'},
                    {'skill': 'place', 'object': 'red_cube', 'zone': 'zone_b'},
                    {'skill': 'home'},
                ],
            },
        }
        body = json.dumps({
            'model': self.model,
            'temperature': 0,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)},
            ],
        }).encode('utf-8')
        request = Request(
            self.base_url + '/chat/completions',
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST',
        )
        with urlopen(request, timeout=18) as response:
            result = json.loads(response.read().decode('utf-8'))
        content = result['choices'][0]['message']['content']
        if isinstance(content, list):
            content = ''.join(part.get('text', '') for part in content)
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content):
        text = content.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)
        return json.loads(text)

    @staticmethod
    def validate_core_plan(plan):
        if not isinstance(plan, dict) or set(plan) != {'plan'}:
            raise ValueError('Core plan must contain only the plan field.')
        steps = plan['plan']
        if not isinstance(steps, list) or len(steps) != 3:
            raise ValueError('Core plan must contain pick, place, and home.')
        pick, place, home = steps
        if not isinstance(pick, dict) or set(pick) != {'skill', 'object'}:
            raise ValueError('Invalid pick step.')
        if not isinstance(place, dict) or set(place) != {'skill', 'object', 'zone'}:
            raise ValueError('Invalid place step.')
        if not isinstance(home, dict) or set(home) != {'skill'}:
            raise ValueError('Invalid home step.')
        if pick.get('skill') != 'pick' or place.get('skill') != 'place' or home.get('skill') != 'home':
            raise ValueError('Core plan skills must be pick, place, home in order.')
        if pick.get('object') not in OBJECTS or place.get('object') != pick.get('object'):
            raise ValueError('Pick and place must name the same known object.')
        if place.get('zone') not in ZONES:
            raise ValueError('Place step must name zone_a, zone_b, or zone_c.')

    @staticmethod
    def _parse_locally(command):
        text = command.lower().replace('-', '_')
        object_aliases = {
            'red_cube': ('red_cube', 'red', 'đỏ', 'do'),
            'yellow_cube': ('yellow_cube', 'yellow', 'vàng', 'vang'),
            'blue_cube': ('blue_cube', 'blue', 'xanh dương', 'xanh duong'),
            'green_cube': ('green_cube', 'green', 'xanh lá', 'xanh la'),
            'purple_cube': ('purple_cube', 'purple', 'tím', 'tim'),
        }
        object_name = None
        for candidate, aliases in object_aliases.items():
            if any(alias in text for alias in aliases):
                object_name = candidate
                break
        zone_match = re.search(
            r'(?:zone|vùng|vung)[_\s-]*([abc])\b', text, flags=re.IGNORECASE)
        zone = f'zone_{zone_match.group(1).lower()}' if zone_match else None
        if object_name is None or zone not in ZONES:
            return None
        return {
            'plan': [
                {'skill': 'pick', 'object': object_name},
                {'skill': 'place', 'object': object_name, 'zone': zone},
                {'skill': 'home'},
            ],
        }
