#!/usr/bin/env python3
"""Probe the live Gemini API free-tier rate limit for a model.

There's no API-key-authenticated endpoint that reports your quota directly
(the aistudio.google.com/rate-limit dashboard needs a Google login, not an
API key). Instead, this fires minimal requests back-to-back until Google's
429 RESOURCE_EXHAUSTED error reports the actual limit value, then reports it.
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(os.path.expanduser('~/.env'))

from google import genai  # noqa: E402
from google.genai import errors  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'gemini-flash-latest'
MAX_PROBES = 20


def main():
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    ok_count = 0

    for i in range(1, MAX_PROBES + 1):
        try:
            client.models.generate_content(model=MODEL, contents='hi')
            ok_count += 1
            print(f'  request {i}: ok')
        except errors.ClientError as e:
            if e.code != 429:
                print(f'  request {i}: unexpected error {e.code}: {e}')
                return
            error = e.details['error']
            violations = next(
                d['violations'] for d in error['details']
                if d.get('@type', '').endswith('QuotaFailure'))
            print(f'\nHit a limit after {ok_count} successful requests.')
            for v in violations:
                print(f"  {v['quotaId']} (model={v['quotaDimensions'].get('model', MODEL)}): "
                      f"{v['quotaValue']}")
            print(f"\nMessage: {error['message'].splitlines()[0]}")
            return
        time.sleep(0.5)

    print(f'\nNo limit hit in {MAX_PROBES} requests (~{MAX_PROBES * 0.5:.0f}s) — quota is higher than that.')


if __name__ == '__main__':
    main()
