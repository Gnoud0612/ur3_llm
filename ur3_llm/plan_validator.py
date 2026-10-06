"""Compile camera-aware skill plans and validate them before execution."""

import math


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


class PlanValidator:
    """Deterministic state-aware planning policy and pre-execution validator."""

    def __init__(self, scene):
        self.scene = scene
        self.zone_centers = scene['zones']
        self.zone_radius = float(scene['vision']['zone_radius'])
        self.workspace = scene['workspace']
        self.objects = set(scene['objects'])
        self.zones = set(scene['zones'])

    def compile_plan(self, core_plan, state):
        """Insert obstacle clearing based on the latest camera frame."""
        object_name = core_plan['plan'][0]['object']
        destination = core_plan['plan'][1]['zone']
        if object_name not in state.get('objects', {}):
            raise ValueError(f'Camera không thấy {object_name}.')
        if destination not in state.get('zones', {}):
            raise ValueError(f'Camera chưa có trạng thái của {destination}.')

        occupant = state['zones'][destination].get('occupied_by')
        steps = [{
            'skill': 'check_zone',
            'zone': destination,
            'observed_occupant': occupant,
        }]
        if occupant == object_name:
            steps.append({'skill': 'home'})
            return {'goal': {'object': object_name, 'zone': destination}, 'plan': steps}

        if occupant is not None:
            if occupant not in state.get('objects', {}):
                raise ValueError(f'{destination} báo vật chiếm chỗ không còn thấy: {occupant}.')
            temporary = self.find_free_position(state, excluded_object=occupant)
            if temporary is None:
                raise ValueError('Không tìm được vị trí tạm trống dựa trên ảnh camera.')
            steps.extend([
                {
                    'skill': 'find_free_position',
                    'object': occupant,
                    'position': {'x': temporary[0], 'y': temporary[1]},
                },
                {'skill': 'pick', 'object': occupant},
                {
                    'skill': 'place',
                    'object': occupant,
                    'target': {'type': 'position', 'x': temporary[0], 'y': temporary[1]},
                },
            ])

        steps.extend([
            {'skill': 'pick', 'object': object_name},
            {
                'skill': 'place',
                'object': object_name,
                'target': {'type': 'zone', 'name': destination},
            },
            {'skill': 'home'},
        ])
        return {'goal': {'object': object_name, 'zone': destination}, 'plan': steps}

    def find_free_position(self, state, excluded_object=None):
        """Choose a reachable candidate that the camera sees as unoccupied."""
        objects = state.get('objects', {})
        clearance = float(self.workspace['object_clearance'])
        for candidate in self.scene.get('temporary_candidates', []):
            x, y = map(float, candidate)
            if not self._inside_workspace(x, y):
                continue
            if any(_distance((x, y), center) < self.zone_radius + 0.025
                   for center in self.zone_centers.values()):
                continue
            occupied = any(
                name != excluded_object
                and _distance((x, y), (obj['x'], obj['y'])) < clearance
                for name, obj in objects.items()
            )
            if not occupied:
                return [round(x, 3), round(y, 3)]
        return None

    def validate(self, plan, state):
        """Check schema, camera agreement, workspace limits and task semantics."""
        try:
            self._validate_or_raise(plan, state)
            return True, 'Kế hoạch hợp lệ theo camera và workspace.'
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            return False, str(exc)

    def _validate_or_raise(self, plan, state):
        if not isinstance(plan, dict) or set(plan) != {'goal', 'plan'}:
            raise ValueError('Plan phải có đúng hai trường goal và plan.')
        goal = plan['goal']
        if not isinstance(goal, dict) or set(goal) != {'object', 'zone'}:
            raise ValueError('Goal cần object và zone.')
        object_name = goal['object']
        destination = goal['zone']
        if object_name not in self.objects or destination not in self.zones:
            raise ValueError('Goal tham chiếu object hoặc zone không hợp lệ.')
        if object_name not in state.get('objects', {}):
            raise ValueError(f'Camera không xác nhận {object_name}.')
        if destination not in state.get('zones', {}):
            raise ValueError(f'Camera không xác nhận {destination}.')

        occupant = state['zones'][destination].get('occupied_by')
        involved = {object_name}
        if occupant is not None:
            involved.add(occupant)
        for name in involved:
            observation = state.get('objects', {}).get(name)
            if observation is None:
                raise ValueError(f'Camera không xác nhận object cần thao tác: {name}.')
            self._validate_position(observation['x'], observation['y'])
        steps = plan['plan']
        if not isinstance(steps, list) or not steps:
            raise ValueError('Plan rỗng.')
        self._expect_step(steps[0], 'check_zone', {'zone', 'observed_occupant'})
        if steps[0]['zone'] != destination or steps[0]['observed_occupant'] != occupant:
            raise ValueError('check_zone không khớp ảnh camera hiện tại.')

        if occupant == object_name:
            expected_skills = ['check_zone', 'home']
        elif occupant is None:
            expected_skills = ['check_zone', 'pick', 'place', 'home']
        else:
            if occupant not in self.objects or occupant not in state.get('objects', {}):
                raise ValueError(f'Camera không thấy vật đang chiếm {destination}.')
            expected_skills = [
                'check_zone', 'find_free_position', 'pick', 'place',
                'pick', 'place', 'home',
            ]
        if [step.get('skill') for step in steps] != expected_skills:
            raise ValueError('Chuỗi skill không tuân theo policy xử lý zone có/không bị chiếm.')
        if steps[-1] != {'skill': 'home'}:
            raise ValueError('Bước cuối phải là home và không được thêm trường khác.')

        self._validate_step_shapes(steps, object_name, destination, occupant)
        simulated_positions = {
            name: (float(obj['x']), float(obj['y']))
            for name, obj in state.get('objects', {}).items()
        }
        held = None
        selected_temp = None
        for step in steps:
            skill = step['skill']
            if skill == 'check_zone':
                continue
            if skill == 'find_free_position':
                mover = step['object']
                point = step['position']
                self._validate_position(point['x'], point['y'])
                if mover not in simulated_positions:
                    raise ValueError(f'Không thể dọn {mover}: không có trong trạng thái camera.')
                self._ensure_clear((point['x'], point['y']), simulated_positions, mover)
                self._ensure_not_a_zone((point['x'], point['y']))
                selected_temp = (mover, float(point['x']), float(point['y']))
            elif skill == 'pick':
                mover = step['object']
                if held is not None or mover not in simulated_positions:
                    raise ValueError(f'pick({mover}) không hợp lệ theo trạng thái mô phỏng.')
                if mover == occupant and occupant != object_name and selected_temp is None:
                    raise ValueError('Phải chọn vị trí tạm trước khi dọn vật đang chiếm zone.')
                held = mover
                simulated_positions.pop(mover)
            elif skill == 'place':
                mover = step['object']
                if held != mover:
                    raise ValueError(f'place({mover}) không khớp vật đang được gắp.')
                target = step['target']
                if target['type'] == 'position':
                    if selected_temp is None or selected_temp[0] != mover:
                        raise ValueError('Vị trí tạm không được camera-aware planner chọn.')
                    x, y = float(target['x']), float(target['y'])
                    if _distance((x, y), selected_temp[1:]) > 1e-6:
                        raise ValueError('place không sử dụng vị trí tạm đã kiểm tra.')
                    self._ensure_clear((x, y), simulated_positions)
                    simulated_positions[mover] = (x, y)
                    selected_temp = None
                else:
                    zone = target['name']
                    if mover != object_name or zone != destination:
                        raise ValueError('Chỉ object user yêu cầu được đặt vào destination zone.')
                    center = self.zone_centers[zone]
                    blocker = self._zone_occupant(center, simulated_positions)
                    if blocker is not None:
                        raise ValueError(f'{zone} vẫn bị chiếm bởi {blocker} trong plan.')
                    simulated_positions[mover] = tuple(center)
                held = None

        if held is not None:
            raise ValueError('Kế hoạch kết thúc khi robot vẫn đang giữ vật.')
        final_center = self.zone_centers[destination]
        final_object = simulated_positions.get(object_name)
        if final_object is None or _distance(final_object, final_center) > self.zone_radius:
            raise ValueError(f'Plan không đặt {object_name} vào {destination}.')

    def _validate_step_shapes(self, steps, requested_object, destination, occupant):
        self._expect_step(steps[0], 'check_zone', {'zone', 'observed_occupant'})
        if occupant == requested_object:
            return
        if occupant is None:
            self._expect_step(steps[1], 'pick', {'object'})
            self._expect_step(steps[2], 'place', {'object', 'target'})
            self._expect_zone_target(steps[2]['target'], destination)
            if steps[1]['object'] != requested_object or steps[2]['object'] != requested_object:
                raise ValueError('Pick/place phải tác động đúng object user yêu cầu.')
            return

        mover = occupant
        self._expect_step(steps[1], 'find_free_position', {'object', 'position'})
        self._expect_step(steps[2], 'pick', {'object'})
        self._expect_step(steps[3], 'place', {'object', 'target'})
        self._expect_step(steps[4], 'pick', {'object'})
        self._expect_step(steps[5], 'place', {'object', 'target'})
        if steps[1]['object'] != mover or steps[2]['object'] != mover or steps[3]['object'] != mover:
            raise ValueError('Các bước dọn zone phải xử lý đúng object đang chiếm chỗ.')
        if steps[3]['target'].get('type') != 'position':
            raise ValueError('Vật dọn ra phải được đặt vào vị trí tạm.')
        if steps[4]['object'] != requested_object or steps[5]['object'] != requested_object:
            raise ValueError('Các bước cuối phải xử lý object user yêu cầu.')
        self._expect_zone_target(steps[5]['target'], destination)

    @staticmethod
    def _expect_step(step, skill, fields):
        if not isinstance(step, dict) or set(step) != {'skill'} | set(fields):
            raise ValueError(f'Step {skill} sai schema.')
        if step['skill'] != skill:
            raise ValueError(f'Expected skill {skill}.')

    def _expect_zone_target(self, target, destination):
        if not isinstance(target, dict) or set(target) != {'type', 'name'}:
            raise ValueError('Zone target cần type và name.')
        if target['type'] != 'zone' or target['name'] != destination:
            raise ValueError('Place target không khớp zone người dùng yêu cầu.')

    def _validate_position(self, x, y):
        if not self._inside_workspace(float(x), float(y)):
            raise ValueError('Vị trí nằm ngoài workspace/reach đã cấu hình.')

    def _inside_workspace(self, x, y):
        return (
            self.workspace['x_min'] <= x <= self.workspace['x_max']
            and self.workspace['y_min'] <= y <= self.workspace['y_max']
            and math.hypot(x, y) <= self.workspace['max_reach']
        )

    def _ensure_clear(self, point, positions, excluded=None):
        clearance = float(self.workspace['object_clearance'])
        for name, other in positions.items():
            if name != excluded and _distance(point, other) < clearance:
                raise ValueError(f'Vị trí tạm quá gần {name}; không an toàn để đặt.')

    def _ensure_not_a_zone(self, point):
        for zone, center in self.zone_centers.items():
            if _distance(point, center) < self.zone_radius + 0.025:
                raise ValueError(f'Vị trí tạm chồng lên vùng đặt {zone}.')

    def _zone_occupant(self, center, positions):
        matches = [
            (_distance(center, point), name)
            for name, point in positions.items()
            if _distance(center, point) <= self.zone_radius
        ]
        return min(matches)[1] if matches else None
