"""USD estimates and usage accounting; unknown charges are never zero."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation


class PricingError(ValueError):
    pass


def amount(value) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() and 0 <= number <= Decimal('1000000') else None
    except InvalidOperation:
        return None


def usd(value) -> str:
    number = amount(value)
    return 'неизвестно' if number is None else '$' + format(number, '.8f')


@dataclass(frozen=True)
class Price:
    input: str
    output: str
    cached: str | None = None
    cache_write: str | None = None
    long_input: str | None = None
    long_output: str | None = None
    long_cached: str | None = None
    long_cache_write: str | None = None
    source: str = ''
    updated_at: str = ''

    def __post_init__(self):
        if self.input is None or self.output is None:
            raise PricingError('Input and output prices are required')
        if any(not isinstance(value, str) or len(value) > 1024 or any(c in value for c in '\n\r\x00') for value in (self.source, self.updated_at)):
            raise PricingError('Invalid price provenance')
        for key in ('input', 'output', 'cached', 'cache_write', 'long_input', 'long_output', 'long_cached', 'long_cache_write'):
            value = getattr(self, key)
            if value is not None and amount(value) is None:
                raise PricingError('Invalid model price')

    def as_dict(self):
        return asdict(self)

    def calculate(self, prompt: int, output: int, cached: int = 0, writes: int = 0) -> Decimal | None:
        if any(type(n) is not int or n < 0 for n in (prompt, output, cached, writes)) or cached + writes > prompt:
            return None
        long = prompt > 272000 and self.long_input is not None
        inp, out = (self.long_input, self.long_output) if long else (self.input, self.output)
        cache, write = (self.long_cached, self.long_cache_write) if long else (self.cached, self.cache_write)
        if out is None or (cached and cache is None) or (writes and write is None):
            return None
        return (prompt - cached - writes) * amount(inp) + output * amount(out) + cached * (amount(cache) or Decimal(0)) + writes * (amount(write) or Decimal(0))


def estimate_tokens(messages, tools) -> int:
    # Includes history, tool arguments/results, schemas and an overhead allowance.
    # UTF-8 / 3 is a heuristic, not a provider tokenizer or a billing guarantee.
    data = json.dumps({'messages': messages, 'tools': tools}, ensure_ascii=False).encode('utf-8')
    return max(1, (len(data) + 2) // 3 + 32 * (len(messages) + len(tools)))


def estimate(price: Price | None, messages, tools, max_tokens: int) -> dict:
    prompt = estimate_tokens(messages, tools)
    low = price.calculate(prompt, 0) if price else None
    high = price.calculate(prompt, max_tokens) if price else None
    return {'input_tokens_estimate': prompt, 'output_tokens_limit': max_tokens,
            'usd_input_estimate': str(low) if low is not None else None,
            'usd_with_full_output_estimate': str(high) if high is not None else None,
            'method': 'utf8_bytes/3 + message/schema overhead; output=0..max_tokens',
            'price': price.as_dict() if price else None}


def clean_usage(value) -> dict:
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
              if type(value.get(key)) is int and value[key] >= 0}
    cost = amount(value.get('cost'))
    if cost is not None:
        result['cost'] = str(cost)
    details = value.get('prompt_tokens_details')
    if isinstance(details, dict):
        result['prompt_tokens_details'] = {key: details[key] for key in ('cached_tokens', 'cache_write_tokens')
                                          if type(details.get(key)) is int and details[key] >= 0}
    return result


def actual_cost(provider: str | None, price: Price | None, usage: dict, *, failed=False) -> dict:
    usage = usage if isinstance(usage, dict) else {}
    clean = clean_usage(usage)
    reported = amount(clean.get('cost'))
    if provider == 'openrouter' and reported is not None:
        value, kind = reported, 'reported'
    elif provider in {'ollama', 'scripted'}:
        value, kind = Decimal(0), 'local'
    elif price and not usage.get('_nonstandard_tier') and 'prompt_tokens' in clean and 'completion_tokens' in clean:
        details = clean.get('prompt_tokens_details', {})
        value = price.calculate(clean['prompt_tokens'], clean['completion_tokens'],
                                details.get('cached_tokens', 0), details.get('cache_write_tokens', 0))
        kind = 'calculated' if value is not None else 'unknown'
    else:
        value, kind = None, 'unknown'
    return {'usd': str(value) if value is not None else None, 'kind': kind, 'usage': clean,
            'price': price.as_dict() if price else None}


def cost_summary(events: list, rounds: int | None = None) -> dict:
    records, attempts = {}, set()
    for event in events:
        if event.get('event') == 'model_request' and type(event.get('round')) is int:
            attempts.add(event['round'])
        if event.get('event') != 'model_cost':
            continue
        number = event.get('round')
        if type(number) is not int or number < 1 or event.get('kind') not in {'reported', 'calculated', 'local', 'unknown'}:
            raise PricingError('Invalid saved cost record')
        if number in records and records[number] != event:
            raise PricingError('Conflicting saved cost records')
        records[number] = event
        attempts.add(number)
    if rounds is not None:
        attempts.update(range(1, rounds + 1))
    totals = dict.fromkeys(('reported', 'calculated', 'local'), Decimal(0))
    unknown = 0
    for number in attempts:
        record = records.get(number)
        value = amount(record.get('usd')) if record else None
        if not record or record['kind'] == 'unknown' or value is None:
            unknown += 1
        else:
            totals[record['kind']] += value
    return {'currency': 'USD', 'known_usd': str(sum(totals.values())),
            'reported_usd': str(totals['reported']), 'calculated_usd': str(totals['calculated']),
            'requests': len(attempts), 'unknown_requests': unknown,
            'complete': unknown == 0, 'calculated_requests': sum(r['kind'] == 'calculated' for r in records.values())}
