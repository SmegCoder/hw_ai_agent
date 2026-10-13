"""Public provider prices and installed Ollama models, with a dated disk cache."""
from __future__ import annotations

import json
import os
import re
import ssl
import stat
import tempfile
import time
import unicodedata
from http.client import HTTPException
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.request import HTTPSHandler, Request, build_opener
from urllib.error import HTTPError

import certifi

from .client import NoRedirect
from .pricing import Price, PricingError, amount, usd

OPENROUTER_MODELS = 'https://openrouter.ai/api/v1/models'
OPENAI_PRICING = 'https://developers.openai.com/api/docs/pricing.md'


@dataclass(frozen=True)
class Model:
    id: str
    price: Price | None
    context: int | None = None
    max_output: int | None = None
    tools: bool = False

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r'[\w./:@+\-]{1,200}', self.id):
            raise PricingError('Invalid catalog model ID')
        for value in (self.context, self.max_output):
            if value is not None and (type(value) is not int or value < 1):
                raise PricingError('Invalid model limits')
        if type(self.tools) is not bool:
            raise PricingError('Invalid tool capability')

    @classmethod
    def restore(cls, data):
        return cls(**{**data, 'price': Price(**data['price']) if data.get('price') else None})


def openrouter_models(body: dict, updated: str) -> list[Model]:
    if not isinstance(body, dict) or not isinstance(body.get('data'), list):
        raise PricingError('Invalid OpenRouter catalog')
    models = []
    for row in body['data'][:5000]:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str):
            continue
        try:
            rates = row.get('pricing', {})
            price = None
            if amount(rates.get('prompt')) is not None and amount(rates.get('completion')) is not None:
                # OpenRouter prices are already USD per token (not per million).
                price = Price(rates['prompt'], rates['completion'], rates.get('input_cache_read'),
                              rates.get('input_cache_write'), source=OPENROUTER_MODELS, updated_at=updated)
            top = row.get('top_provider') or {}
            text_output = 'text' in row.get('architecture', {}).get('output_modalities', ['text'])
            models.append(Model(row['id'], price, row.get('context_length') or None,
                                top.get('max_completion_tokens') or None,
                                'tools' in row.get('supported_parameters', []) and text_output and not row['id'].endswith(':batch')))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    if not models:
        raise PricingError('OpenRouter catalog has no valid models')
    return models


def openai_models(markdown: str, updated: str) -> list[Model]:
    # Only the explicit Standard table; never Batch/Flex/Fast prices.
    section = markdown.partition('### Standard pricing data')[2].split('### Batch pricing data')[0]
    if not section:
        raise PricingError('Official OpenAI Standard pricing table not found')
    models = []
    for line in section.splitlines():
        if not line.startswith('| '):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) != 9 or not cells[1].startswith('$'):
            continue
        identifier = cells[0].split(' (')[0]
        try:
            rates = [str(amount(cell.removeprefix('$')) / Decimal(1_000_000))
                     if amount(cell.removeprefix('$')) is not None else None for cell in cells[1:]]
            if rates[0] is None or rates[3] is None:
                continue
            price = Price(rates[0], rates[3], rates[1], rates[2], rates[4], rates[7], rates[5], rates[6],
                          source=OPENAI_PRICING, updated_at=updated)
            # Pricing is public; account access isn't established by this table.
            # Restrict the picker/router to known Chat Completions tool families.
            tools = identifier.startswith(('gpt-4.1', 'gpt-4o', 'gpt-5')) and '-pro' not in identifier
            models.append(Model(identifier, price, tools=tools))
        except (TypeError, ValueError):
            continue
    if not models:
        raise PricingError('Official OpenAI table has no valid prices')
    return models


class ModelCatalog:
    def __init__(self, provider: str, cache_dir: Path, *, base_url=None, ttl=3600, timeout=8):
        self.provider, self.cache_dir = provider, Path(cache_dir)
        self.base_url, self.ttl, self.timeout = base_url, ttl, timeout
        self.updated_at, self.status, self.warning = '', '', None
        self.source = OPENROUTER_MODELS if provider == 'openrouter' else OPENAI_PRICING if provider == 'openai' else base_url
        self.models = []
        self.opener = build_opener(NoRedirect(), HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())))

    def _get(self, url, data=None):
        request = Request(url, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type': 'application/json', 'User-Agent': 'hw-ai-agent/0.1'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(5_000_001)
            if len(raw) > 5_000_000:
                raise PricingError('Catalog response is too large')
            return raw.decode('utf-8')
        except HTTPError as exc:
            exc.close()
            raise PricingError(f'Каталог {self.provider} вернул HTTP {exc.code}') from None
        except (OSError, ValueError, HTTPException) as exc:
            raise PricingError(f'Каталог {self.provider} недоступен ({type(exc).__name__})') from None

    def _fetch(self):
        updated = datetime.now(timezone.utc).isoformat()
        if self.provider == 'openrouter':
            models = openrouter_models(json.loads(self._get(OPENROUTER_MODELS)), updated)
        elif self.provider == 'openai':
            models = openai_models(self._get(OPENAI_PRICING), updated)
        elif self.provider == 'ollama':
            if not self.base_url:
                raise PricingError('Ollama base URL is required')
            native = self.base_url.removesuffix('/v1')
            rows = json.loads(self._get(native + '/api/tags'))['models']
            models = []
            for row in rows[:50]:
                name = row['name']
                model = Model(name, Price('0', '0', source=native, updated_at=updated))
                try:
                    info = json.loads(self._get(native + '/api/show', {'model': name}))
                    contexts = [value for key, value in info.get('model_info', {}).items() if key.endswith('.context_length') and type(value) is int and value > 0]
                    model = Model(name, model.price, max(contexts) if contexts else None,
                                  tools='tools' in info.get('capabilities', []))
                except (PricingError, ValueError, KeyError, TypeError, AttributeError):
                    pass
                models.append(model)
            if not models:
                raise PricingError('В Ollama нет загруженных моделей')
        else:
            raise PricingError('Unsupported model catalog provider')
        return models, updated

    def load(self, *, refresh=False):
        path = self.cache_dir / f'models-{self.provider}.json'
        cached = None
        try:
            if path.is_symlink():
                raise ValueError
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, 'rb') as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError
                raw = source.read(5_000_001)
            if len(raw) > 5_000_000:
                raise ValueError
            data = json.loads(raw)
            age = time.time() - data['fetched_at']
            if data['source'] != self.source or not -300 <= age:
                raise ValueError
            cached = ([Model.restore({**m, 'tools': m['tools'] and not m['id'].endswith(':batch')}) for m in data['models'][:5000]], data['updated_at'], age)
            if not cached[0]:
                raise ValueError
        except (OSError, KeyError, TypeError, ValueError, AttributeError):
            cached = None
        if cached and cached[2] <= self.ttl and not refresh:
            self.models, self.updated_at, self.status = cached[0], cached[1], 'cache'
            return self
        try:
            self.models, self.updated_at = self._fetch()
            self.status = 'live'
        except (PricingError, ValueError, KeyError, TypeError, AttributeError):
            if not cached:
                raise PricingError(f'Не удалось получить каталог {self.provider}; неизвестные цены не считаются нулевыми') from None
            self.models, self.updated_at, self.status = cached[0], cached[1], 'stale'
            self.warning = 'Сеть недоступна: используются устаревшие цены из кэша.'
            return self
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor, temporary = tempfile.mkstemp(prefix='models-', dir=self.cache_dir)
            try:
                with os.fdopen(descriptor, 'w', encoding='utf-8') as target:
                    json.dump({'source': self.source, 'fetched_at': time.time(), 'updated_at': self.updated_at,
                               'models': [asdict(m) for m in self.models]}, target, ensure_ascii=False, allow_nan=False)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        except OSError:
            self.warning = 'Тарифы получены, но кэш сохранить не удалось.'
        return self

    def find(self, identifier):
        return next((m for m in self.models if m.id == identifier), None)


def comparison(models, input_tokens=1000, output_tokens=1000, *, numbered=False) -> str:
    headers = ['Модель', 'вход $/1M', 'выход $/1M', 'контекст', 'tools', 'сценарий, USD']
    rows = []
    def decimal_text(value):
        text = format(value, 'f')
        return text.rstrip('0').rstrip('.') if '.' in text else text
    for index, model in enumerate(models, 1):
        price = model.price
        inp = decimal_text(amount(price.input) * 1_000_000) if price else '?'
        out = decimal_text(amount(price.output) * 1_000_000) if price else '?'
        total = price.calculate(input_tokens, output_tokens) if price else None
        context = format(model.context, ',').replace(',', ' ') if model.context else '?'
        row = [model.id, inp, out, context, 'да' if model.tools else '?', usd(total)]
        rows.append([str(index), *row] if numbered else row)
    if numbered:
        headers.insert(0, '№')
    def width(text):
        return sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in {'W', 'F'} else 1 for char in text)
    widths = [max(width(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    numeric = {1, 2, 3, 5} if not numbered else {0, 2, 3, 4, 6}
    def render(row):
        return ' | '.join((' ' * (widths[column] - width(cell)) + cell) if column in numeric
                          else (cell + ' ' * (widths[column] - width(cell))) for column, cell in enumerate(row))
    lines = [render(headers), '-+-'.join('-' * size for size in widths), *(render(row) for row in rows)]
    if any(not model.tools for model in models):
        lines.append('tools ?: поддержка инструментов не подтверждена каталогом.')
    return '\n'.join(lines)
