#!/usr/bin/env python3
"""Provision a Twilio phone number for Jeff's Signal registration.

Jeff's Signal identity is a phone number, and it needs a *dedicated* one (not a
personal number). This helper drives the Twilio REST API to (1) search for an
SMS-capable number and (2) — only when you explicitly ask — buy one. The number
just has to receive the one-time Signal verification code; ongoing Signal traffic
never flows back through Twilio.

Safety: **search is the default. Nothing is purchased unless you pass `--buy
<E.164>` with a specific number from the search results.** So a bare run can
never spend money — you look, then you choose.

Stdlib only (urllib) so it runs without Jeff's virtualenv. Twilio credentials are
read from the environment — never hard-coded, never committed:

    TWILIO_ACCOUNT_SID   your Twilio Account SID (starts AC…)
    TWILIO_AUTH_TOKEN    your Twilio Auth Token

Populate them from your secret store at run time, e.g. with the 1Password CLI:

    export TWILIO_ACCOUNT_SID="$(op read 'op://<vault>/jeff-signal-twilio/account-sid')"
    export TWILIO_AUTH_TOKEN="$(op read 'op://<vault>/jeff-signal-twilio/auth-token')"

(Replace <vault> with your own vault name — kept generic here because this repo
mirrors to a public remote.)

Usage:
    # 1. Search for SMS-capable US numbers in area code 415:
    python scripts/provision_signal_number.py --country US --area-code 415

    # 2. Buy a specific one from the results (this DOES cost a small monthly fee):
    python scripts/provision_signal_number.py --buy +14155550123

    # 3. Store the number back in your secret store (the script prints the exact
    #    command; nothing is written to 1Password by this script).

See docs/signal-provisioning.md for the full runbook (Signal registration,
the registration-lock PIN, and how the number is wired into Jeff).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

_API = "https://api.twilio.com/2010-04-01"


def _auth_header() -> str:
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if not sid or not token:
        sys.exit(
            "error: set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in the environment "
            "(see the module docstring for an `op read` example)."
        )
    raw = f"{sid}:{token}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _request(method: str, url: str, *, data: dict | None = None) -> dict:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", _auth_header())
    if body is not None:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        sys.exit(f"error: Twilio API {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"error: could not reach Twilio: {exc.reason}")


def _account_sid() -> str:
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    if not sid:
        sys.exit("error: set TWILIO_ACCOUNT_SID in the environment (see the module docstring).")
    return sid


def search(country: str, area_code: str | None, limit: int) -> None:
    """List SMS-capable numbers. Some VoIP ranges are flagged by Signal, so this
    filters to SMS-enabled numbers; if registration is later rejected, fall back
    to voice verification (see the runbook)."""
    params: dict[str, str] = {"SmsEnabled": "true", "PageSize": str(limit)}
    if area_code:
        params["AreaCode"] = area_code
    qs = urllib.parse.urlencode(params)
    url = f"{_API}/Accounts/{_account_sid()}/AvailablePhoneNumbers/{country}/Local.json?{qs}"
    result = _request("GET", url)
    numbers = result.get("available_phone_numbers", [])
    if not numbers:
        print("No SMS-capable numbers found for those criteria. Try another area code/country.")
        return
    print(f"Found {len(numbers)} SMS-capable number(s):\n")
    for n in numbers:
        caps = n.get("capabilities", {})
        flags = ",".join(k for k, v in caps.items() if v)
        print(f"  {n.get('phone_number'):16}  {n.get('friendly_name', ''):20}  [{flags}]")
    print(
        "\nPick one and buy it with:\n"
        f"  python {sys.argv[0]} --buy <PHONE_NUMBER>"
    )


def buy(phone: str) -> None:
    """Purchase a specific number. This is the only money-spending action and it
    requires an explicit E.164 number — never a wildcard."""
    url = f"{_API}/Accounts/{_account_sid()}/IncomingPhoneNumbers.json"
    result = _request("POST", url, data={"PhoneNumber": phone})
    bought = result.get("phone_number", phone)
    print(f"Purchased {bought}\n")
    print("Next steps:")
    print("  1. Store it in your secret store, e.g.:")
    print(
        f"       op item edit 'jeff-signal-twilio' 'number={bought}' --vault '<vault>'"
    )
    print("  2. Register it with Signal + set the registration-lock PIN —")
    print("     see docs/signal-provisioning.md.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Provision a Twilio number for Jeff's Signal registration "
        "(search by default; --buy to purchase)."
    )
    parser.add_argument("--country", default="US", help="ISO country code (default US)")
    parser.add_argument("--area-code", default=None, help="Preferred area code (optional)")
    parser.add_argument("--limit", type=int, default=10, help="Max results to list (default 10)")
    parser.add_argument(
        "--buy",
        metavar="E164",
        default=None,
        help="Buy this exact number (E.164, e.g. +14155550123). Spends money.",
    )
    args = parser.parse_args(argv)

    if args.buy:
        buy(args.buy)
    else:
        search(args.country, args.area_code, args.limit)


if __name__ == "__main__":
    main()
