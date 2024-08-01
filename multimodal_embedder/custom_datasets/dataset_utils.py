from copy import deepcopy


def format_task(task, params):
    task = deepcopy(task)
    for key, value in task.items():
        if isinstance(value, str):
            task[key] = value.format(**params)
        elif isinstance(value, list):
            task[key] = [format_task(v, params) for v in value]
        elif isinstance(value, dict):
            task[key] = format_task(value, params)
    return task