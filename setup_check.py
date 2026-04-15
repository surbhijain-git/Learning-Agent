from dotenv import load_dotenv
import os

load_dotenv()

REQUIRED = [
    "ANTHROPIC_API_KEY",
    "NOTION_TOKEN",
    "NEWSLETTER_EMAIL",
    "NEWSLETTER_EMAIL_PASSWORD",
]

OPTIONAL = [
    "NOTION_DB_ID",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "READWISE_TOKEN",
]

PLACEHOLDERS = {
    "PASTE_YOUR_KEY_HERE",
    "PASTE_YOUR_NOTION_TOKEN_HERE",
    "PASTE_YOUR_DEDICATED_GMAIL_HERE",
    "PASTE_YOUR_APP_PASSWORD_HERE",
    "PASTE_YOUR_MAIN_GMAIL_HERE",
    "FILL_IN_AFTER_STEP_1",
    "FILL_IN_LATER",
}

missing = 0

print("--- Required keys ---")
for key in REQUIRED:
    val = os.getenv(key, "")
    if val and val not in PLACEHOLDERS:
        print(f"  ✅  {key}")
    else:
        print(f"  ❌  {key}")
        missing += 1

print("\n--- Optional keys ---")
for key in OPTIONAL:
    val = os.getenv(key, "")
    if val and val not in PLACEHOLDERS:
        print(f"  ✅  {key}")
    else:
        print(f"  ⏳  {key}  (fill in later)")

print()
if missing == 0:
    print("Ready to build.")
else:
    print(f"{missing} key{'s' if missing != 1 else ''} missing — fill in .env first.")
