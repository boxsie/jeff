# Signal provisioning runbook

How to give Jeff its own Signal number so you can text it like a person and get
LLM replies on the same thread. This is the **operator** side of the Signal front
door: a few real-world steps (a paid number, a verification code, a secret store)
that can't be automated in code. The code half — the in-process Signal client and
front-door loop — already ships in Jeff (`jeff/signal_cli.py`,
`jeff/signal_front.py`); this runbook is what stands it up.

> Placeholders: replace `<vault>` with your own secret-store vault name and
> `+1XXXXXXXXXX` with the real number. Names are kept generic because this repo
> mirrors to a public remote — don't commit real numbers, vault names, or hosts.

## What you'll end up with

- A dedicated Twilio number (Jeff's Signal identity).
- That number registered with Signal as **primary** (not linked to a personal
  phone — a linked device can read the whole account).
- Credentials in your secret store; Jeff configured via `JEFF_SIGNAL_*` env.
- Texting the number from an allowlisted phone → an LLM reply on the same thread.

---

## 1. Twilio account + credentials

1. Create a Twilio account and note the **Account SID** (`AC…`) and **Auth Token**.
2. Store them in your secret store, e.g. with the 1Password CLI:

   ```bash
   op item create --category 'API Credential' --vault '<vault>' \
     --title 'jeff-signal-twilio' \
     'account-sid=AC...' 'auth-token=...'
   ```

## 2. Provision an SMS-capable number

Some Twilio/VoIP ranges are flagged by Signal, so prefer an SMS-capable number
and be ready to fall back to **voice** verification.

```bash
export TWILIO_ACCOUNT_SID="$(op read 'op://<vault>/jeff-signal-twilio/account-sid')"
export TWILIO_AUTH_TOKEN="$(op read 'op://<vault>/jeff-signal-twilio/auth-token')"

# Search (no charge):
python scripts/provision_signal_number.py --country US --area-code 415

# Buy a specific one from the results (small monthly fee):
python scripts/provision_signal_number.py --buy +1XXXXXXXXXX
```

The script **only searches** unless you pass `--buy <E.164>`, so a bare run can't
spend money.

## 3. Store the number

```bash
op item edit 'jeff-signal-twilio' 'number=+1XXXXXXXXXX' --vault '<vault>'
```

Keep the Twilio number **alive** (don't let it lapse/recycle) — losing it means
losing Jeff's Signal identity. It only needs to *receive* the one-time code; no
ongoing Signal traffic flows through Twilio.

## 4. Stand up signal-cli-rest-api

Run [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) in
**`json-rpc` / native mode** (the `normal` mode does not behave for receive).
Persist its data dir on a **volume** — after registration it holds the Signal
session/identity keys, a top-tier secret, and losing it drops the registration.

```yaml
# sketch — fold into Jeff's existing deployment (a sidecar + a volume + a secret)
environment:
  MODE: json-rpc
volumes:
  - signal-cli-data:/home/.local/share/signal-cli   # persistent
```

## 5. Register the number with Signal

Signal requires a CAPTCHA token — solve one at
<https://signalcaptchas.org/registration/generate.html> and copy the
`signalcaptcha://…` token.

```bash
# Request the code (SMS by default):
curl -X POST 'http://signal-cli:8080/v1/register/+1XXXXXXXXXX' \
     -H 'Content-Type: application/json' \
     -d '{"captcha": "<token>", "use_voice": false}'

# If SMS is rejected (VoIP flagged), retry with voice:
#   -d '{"captcha": "<token>", "use_voice": true}'

# Confirm with the code that arrives at the number:
curl -X POST 'http://signal-cli:8080/v1/register/+1XXXXXXXXXX/verify/123-456'
```

## 6. Set the registration-lock PIN

So the number can't be re-registered out from under Jeff:

```bash
curl -X POST 'http://signal-cli:8080/v1/configuration/+1XXXXXXXXXX/settings' \
     -H 'Content-Type: application/json' \
     -d '{"registration_lock": "<pin>"}'
```

Store the PIN alongside the other creds (`op item edit … 'registration-pin=<pin>'`).

## 7. Configure Jeff

Set these on Jeff's deployment (the secret/volume live in your deploy repo, the
same way the existing admin key does — not in this repo):

| Env | Value |
| --- | --- |
| `JEFF_SIGNAL_ENABLED` | `true` |
| `JEFF_SIGNAL_API_URL` | the signal-cli-rest-api base URL (e.g. `http://signal-cli:8080`) |
| `JEFF_SIGNAL_NUMBER` | `+1XXXXXXXXXX` (Jeff's number) |
| `JEFF_SIGNAL_ALLOWLIST` | your phone number(s), comma-separated E.164 — **empty = answer nobody** |
| `JEFF_SIGNAL_POLL_INTERVAL` | `1.0` (optional) |

## 8. Verify

- Text Jeff's number from an **allowlisted** phone → you get an LLM reply on the
  same thread.
- Text from a non-allowlisted number → silence (default-deny).
- Restart Jeff → the registration survives (it lives on the signal-cli volume,
  not in Jeff).

---

## Notes

- Jeff now holds **Signal credentials** — a real secret, separate from any
  Ensemble key. Treat the signal-cli data dir + the PIN like the admin key.
- The number, Twilio creds, and PIN all belong in your secret store, never in
  this repo or in env committed to git.
