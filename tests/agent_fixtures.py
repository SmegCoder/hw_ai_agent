import json


def tool_call(identifier, name, arguments):
    return {"id": identifier, "function": {"name": name, "arguments": json.dumps(arguments)}}


PLAN = {'goal': 'Исправить расчёт стоимости заказа без изменения тестов',
             'steps': [{'description': 'Исправить формулу и проверку диапазона скидки', 'tools': ['write_file']},
                       {'description': 'Запустить контрольные тесты', 'tools': ['Bash']}],
             'risks': ['Не изменять контрольный тестовый файл'],
             'assumptions': ['Допустимая скидка — от 0 до 100 процентов'],
             'done_criteria': ['Шесть тестов проходят', 'Тестовый файл не изменён']}


TASK_PLAN = {'goal': 'Исправить расчёт чека с налогом 20%', 'steps': [
    {'id': 'tax', 'description': 'Исправить процентный налог', 'tools': ['read_file', 'write_file'], 'depends_on': [],
     'verification': {'kind': 'file_written', 'path': 'tax.py'}},
    {'id': 'receipt', 'description': 'Исправить суммирование цен в чеке', 'tools': ['read_file', 'write_file'], 'depends_on': ['tax'],
     'verification': {'kind': 'file_written', 'path': 'receipt.py'}},
    {'id': 'tests', 'description': 'Проверить оба исправления тестами', 'tools': ['Bash'], 'depends_on': ['tax', 'receipt'],
     'verification': {'kind': 'tests_passed', 'command': 'python -m unittest discover -v', 'cwd': '.'}}
], 'risks': ['Тесты нельзя изменять'], 'assumptions': ['Налог составляет 20%'],
'done_criteria': ['Четыре контрольных теста проходят', 'Исходные тесты не изменены']}
