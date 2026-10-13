from __future__ import annotations

import difflib
import json
import sys
from .planning import PlanApprovalDeferred
from .session import SessionError
from .tools import ToolError


def visible(value: str) -> str:
    return ''.join(c if c.isprintable() or c in '\n\t' else repr(c)[1:-1] for c in str(value))


def confirm_plan(plan: dict) -> bool:
    print('\nПлан работы:')
    print('Цель: ' + visible(plan['goal']))
    for number, step in enumerate(plan['steps'], 1):
        print(f"{number}. {visible(step['description'])}\n   Инструменты: {', '.join(step['tools'])}")
        if 'id' in step:
            print('   Задача: ' + step['id'] + '; зависит от: ' + (', '.join(step['depends_on']) or '—'))
            verification = step['verification']
            descriptions = {'file_written': 'Файл соответствует выполненной записи', 'file_read': 'Прочитано актуальное содержимое файла', 'file_absent': 'Файл отсутствует после удаления', 'tests_passed': 'Контрольные тесты проходят'}
            target = verification.get('path', verification.get('command', ''))
            print('   Проверка: ' + descriptions[verification['kind']] + ': ' + visible(target))
    for key, title in (('risks', 'Риски'), ('assumptions', 'Допущения'), ('done_criteria', 'Критерии готовности')):
        print(title + ':')
        for item in plan[key]:
            print('  - ' + visible(item))
        if not plan[key]:
            print('  (не указаны)')
    if not sys.stdin.isatty():
        raise PlanApprovalDeferred('План сохранён; запустите resume в терминале для утверждения')
    try:
        while True:
            answer = input('Утвердить план и перейти к исполнению? [y/n; Enter — пауза]: ').strip().lower()
            if answer in {'y', 'yes', 'да', 'д'}:
                return True
            if answer in {'n', 'no', 'нет', 'н'}:
                return False
            if not answer:
                raise PlanApprovalDeferred('План ожидает решения; продолжение через resume')
            print('Введите y для утверждения или n для отклонения.')
    except EOFError:
        raise PlanApprovalDeferred('Ответ не получен; план сохранён без утверждения') from None


def choose_session(store, identifier: str | None, *, include_completed=False, journal_only=False) -> str:
    if identifier is not None:
        return identifier
    if not sys.stdin.isatty():
        raise SessionError('Без интерактивного терминала укажите SESSION_ID явно')
    candidates = [s for s in store.list_sessions() if (include_completed or s['status'] not in {'completed', 'aborted','rolled_back'}) and (not journal_only or store.get(s['id']).metadata.get('journal_mode',False))][:20]
    if not candidates:
        raise SessionError('Нет незавершённых сессий для продолжения')
    if len(candidates) == 1:
        print('Продолжаем сессию: ' + candidates[0]['id'])
        return candidates[0]['id']
    print('\nВыберите сессию для продолжения:')
    for number, item in enumerate(candidates, 1):
        print(f"{number}. {item['status']} | {visible(item['model'] or 'offline')} | {visible(item['workspace'])}")
    try:
        while True:
            value = input('Номер сессии (Enter — отмена): ').strip()
            if not value:
                raise SessionError('Выбор сессии отменён')
            if value.isascii() and value.isdigit() and 1 <= int(value) <= len(candidates):
                return candidates[int(value) - 1]['id']
            print('Введите номер из списка.')
    except EOFError:
        raise SessionError('Выбор сессии не получен') from None


def confirm_conflict(runtime, item: dict, verify=None) -> str | None:
    if not sys.stdin.isatty():
        return None
    function = item['call']['function']
    print(f"\nНеизвестен результат вызова {item['key']}: {visible(function['name'])}")
    try:
        arguments = json.loads(function['arguments'])
        if not isinstance(arguments, dict):
            raise ValueError('arguments are not an object')
        print('Предложенное действие: ' + visible(function['arguments'])[:4000])
        if 'path' in arguments:
            path = runtime._safe_path(arguments['path'])
            if path.exists():
                current = runtime.read_file(arguments['path'])
                print('Текущее содержимое:\n' + visible(current[:4000]))
            else:
                current = ''
                print('Файл сейчас отсутствует.')
            if function['name'] == 'write_file' and isinstance(arguments.get('content'), str):
                diff = ''.join(difflib.unified_diff(current.splitlines(True), arguments['content'].splitlines(True),
                                                  fromfile='current', tofile='proposed'))
                print('Сравнение с предложенной записью:\n' + visible(diff[:4000] or '(содержимое совпадает)'))
    except (ToolError, OSError, ValueError, TypeError):
        print('Текущее состояние нельзя безопасно показать; проверьте workspace вручную.')
    supported = function['name'] in {'write_file', 'delete_file', 'read_file', 'list_files', 'Bash', 'bash', 'submit_plan', 'complete_task', 'fail_task'}
    print('1 — повторить поддержанный инструмент' if supported else 'Повтор этого инструмента не поддерживается.')
    print('2 — считать вызов неуспешным и продолжить\n3 — завершить с ошибкой')
    if function['name'] == 'write_file':
        print('4 — считать запись выполненной (требуется точное совпадение байтов)')
    print('0 — оставить сессию заблокированной')
    try:
        while True:
            choice = input('Ваш выбор: ').strip()
            if choice in {'0', ''}:
                return None
            if choice == '1' and supported:
                return 'retry'
            if choice == '2':
                return 'skip'
            if choice == '3':
                return 'abort'
            if choice == '4' and function['name'] == 'write_file':
                if verify is not None and verify(item) is True:
                    return 'applied'
                print('Текущее содержимое не подтверждает запись; выберите другое решение.')
                continue
            print('Введите номер доступного действия.')
    except EOFError:
        return None


def confirm_attribution(view: dict) -> str | None:
    print('\nИзменения, обнаруженные после сбоя:\n' + visible(view['diff']))
    print('1 — отнести все показанные изменения к этой сессии')
    print('2 — считать их внешними; не включать в откат сессии')
    print('0 — оставить вызов заблокированным')
    if not sys.stdin.isatty():
        return None
    try:
        while True:
            answer=input('Принадлежность изменений: ').strip()
            if answer=='1':return 'own'
            if answer=='2':return 'external'
            if answer in {'0',''}:return None
            print('Введите 1, 2 или 0.')
    except EOFError:
        return None
