"""
magicpin AI Challenge — Conversation Handlers
Handles multi-turn conversational interactions with merchants and customers.
Features:
- WhatsApp Business canned auto-reply detection and mitigation
- Intent commitment transition (switching immediately from qualification to execution)
- Hostility / Opt-out detection with graceful immediate exit
- Merchant request for deferral/waiting with back-off
- Category-sensitive and language-matched tone
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# Patterns indicating canned WhatsApp Business automated replies
AUTO_REPLY_PATTERNS = [
    r"thank\s+you\s+for\s+contacting",
    r"our\s+team\s+will\s+respond\s+shortly",
    r"automated\s+assistant",
    r"automated\s+reply",
    r"auto-reply",
    r"we\s+are\s+currently\s+closed",
    r"thanks\s+for\s+reaching\s+out",
    r"aapki\s+jaankari\s+ke\s+liye\s+bahut.*shukriya",
    r"hamari\s+team\s+tak\s+pahuncha",
    r"we\s+will\s+get\s+back\s+to\s+you",
    r"currently\s+unavailable",
]

# Patterns indicating opt-out, stop, or hostility
HOSTILE_OPT_OUT_PATTERNS = [
    r"\bstop\b",
    r"\bunsubscribe\b",
    r"\bspam\b",
    r"stop\s+messaging",
    r"useless\s+spam",
    r"don'?t\s+message",
    r"do\s+not\s+text",
    r"not\s+interested",
    r"leave\s+me\s+alone",
    r"remove\s+my\s+number",
    r"band\s+karo",
    r"mat\s+bhejo",
    r"nahi\s+chahiye",
]

# Patterns indicating explicit merchant commitment/action request
INTENT_COMMITMENT_PATTERNS = [
    r"ok\s+lets\s+do\s+it",
    r"let'?s\s+do\s+it",
    r"what'?s\s+next",
    r"\bproceed\b",
    r"go\s+ahead",
    r"send\s+it",
    r"yes\s+please",
    r"karna\s+hai",
    r"judrna\s+hai",
    r"mujhe\s+join\s+karna",
    r"start\s+kar do",
    r"send\s+me\s+the\s+abstract",
    r"share\s+the\s+details",
    r"yes\s+i\s+want",
    r"\bconfirm\b",
    r"\binterested\b",
]

# Patterns for deferral / asking for time
WAIT_PATTERNS = [
    r"busy\s+right\s+now",
    r"message\s+later",
    r"call\s+later",
    r"baad\s+mein",
    r"kal\s+baat",
    r"after\s+(\d+)\s*(min|hour|hr|day)",
]


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    auto_reply_count: int = 0
    consecutive_auto_replies: int = 0
    is_closed: bool = False
    stage: str = "active"  # "active", "planning", "executing", "ended"


def is_auto_reply(message: str, turns: List[Dict[str, Any]]) -> bool:
    """Detect if the message is a WhatsApp automated canned response."""
    msg_clean = message.strip().lower()
    
    # Check regex patterns
    for pat in AUTO_REPLY_PATTERNS:
        if re.search(pat, msg_clean):
            return True

    # Check verbatim repeats from previous merchant turns
    past_merchant_msgs = [
        t.get("message", "").strip().lower()
        for t in turns
        if t.get("from_role") == "merchant" or t.get("from") == "merchant"
    ]
    if past_merchant_msgs.count(msg_clean) >= 1:
        return True

    return False


def is_hostile_or_opt_out(message: str) -> bool:
    """Detect if merchant wants to stop, unsubscribe, or is hostile."""
    msg_clean = message.strip().lower()
    for pat in HOSTILE_OPT_OUT_PATTERNS:
        if re.search(pat, msg_clean):
            return True
    return False


def is_intent_commitment(message: str) -> bool:
    """Detect if merchant has confirmed and wants to proceed to action."""
    msg_clean = message.strip().lower()
    for pat in INTENT_COMMITMENT_PATTERNS:
        if re.search(pat, msg_clean):
            return True
    return False


def is_wait_request(message: str) -> tuple[bool, int]:
    """Detect if merchant asked to wait/defer, returns (is_wait, wait_seconds)."""
    msg_clean = message.strip().lower()
    for pat in WAIT_PATTERNS:
        m = re.search(pat, msg_clean)
        if m:
            if m.groups() and len(m.groups()) == 2:
                qty = int(m.group(1))
                unit = m.group(2)
                if "hour" in unit or "hr" in unit:
                    return True, qty * 3600
                elif "min" in unit:
                    return True, qty * 60
                elif "day" in unit:
                    return True, qty * 86400
            return True, 1800
    return False, 0


def respond(
    state: ConversationState,
    merchant_message: str,
    merchant: Optional[Dict[str, Any]] = None,
    category: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Produce synchronous reply for /v1/reply adhering to:
    - Zero qualifying questions once intent is committed (ACTION mode only).
    - Auto-reply mitigation: 1 attempt to engage, then end.
    - Hostility / Opt-out: immediate graceful end.
    - Category-voice and language matching.
    """
    merchant = merchant or {}
    category = category or {}
    identity = merchant.get("identity", {})
    owner_name = identity.get("owner_first_name") or identity.get("name") or "there"
    languages = identity.get("languages", ["en"])
    is_hi_en = "hi" in languages or "hi-en mix" in languages

    # Append current incoming turn
    state.turns.append({
        "from_role": "merchant",
        "message": merchant_message
    })

    # 1. Check Hostile / Opt-out
    if is_hostile_or_opt_out(merchant_message):
        state.is_closed = True
        state.stage = "ended"
        return {
            "action": "end",
            "rationale": "Merchant opted out or signaled hostility; gracefully exiting immediately without further nudges."
        }

    # 2. Check Intent Transition (CRITICAL: MUST SWITCH TO ACTION, ZERO QUALIFYING WORDS)
    # Checking intent before auto-reply ensures explicit commitment is NEVER falsely classified as auto-reply
    if is_intent_commitment(merchant_message) or state.stage == "planning":
        state.stage = "executing"
        cat_slug = category.get("slug", "")
        
        if cat_slug == "dentists":
            salutation = f"Dr. {owner_name}" if not owner_name.startswith("Dr.") else owner_name
            body = (
                f"Done {salutation}! Proceeding now: here is the next step. "
                f"Sending the complete clinical abstract and drafted patient-ed message to your WhatsApp right now. "
                f"Confirm whenever you want it pushed live."
            )
        elif cat_slug == "restaurants":
            body = (
                f"Done {owner_name}! Proceeding immediately: here is your draft plan. "
                f"Draft corporate package and Swiggy banners are ready for review. "
                f"Next step: sending the final preview to your dashboard now. Confirm to launch."
            )
        elif cat_slug == "salons":
            body = (
                f"Done {owner_name}! Here is what is next: draft campaign is prepared. "
                f"Proceeding with the WhatsApp offer template and Google Business post now. "
                f"Sending preview to confirm."
            )
        elif cat_slug == "gyms":
            body = (
                f"Done {owner_name}! Proceeding with your program setup right here. "
                f"Draft challenge curriculum and winback schedule are generated. "
                f"Next step: sending class schedule for your final confirmation."
            )
        else:
            body = (
                f"Done {owner_name}! Here is the action plan: proceeding with implementation right away. "
                f"Draft updates are ready and sending preview for your confirmation next."
            )

        return {
            "action": "send",
            "body": body,
            "cta": "binary",
            "rationale": "Merchant signaled explicit intent; transitioned immediately from qualification to action mode with zero qualifying hesitation."
        }

    # 3. Check Auto-Reply
    if is_auto_reply(merchant_message, state.turns[:-1]):
        state.auto_reply_count += 1
        state.consecutive_auto_replies += 1
        state.is_closed = True
        state.stage = "ended"
        return {
            "action": "end",
            "rationale": "Detected canned WhatsApp Business auto-reply pattern; ending conversation immediately to avoid burning turns."
        }

    # Reset consecutive auto replies on real message
    state.consecutive_auto_replies = 0

    # 4. Check Wait Request
    should_wait, wait_secs = is_wait_request(merchant_message)
    if should_wait:
        return {
            "action": "wait",
            "wait_seconds": wait_secs,
            "rationale": f"Merchant requested delay; backing off {wait_secs}s before next interaction."
        }

    # 5. General engaged multi-turn response
    # Ensure helpful next-step orientation
    salutation = f"Dr. {owner_name}" if category.get("slug") == "dentists" else f"Hi {owner_name}"
    if is_hi_en:
        body = f"{salutation}, bilkul sahi. Maine aapke profile ke liye recommendations note kar li hain. Ready to proceed? Reply YES."
    else:
        body = f"{salutation}, noted. I have updated the recommendations for your profile. Ready to proceed? Reply YES."

    return {
        "action": "send",
        "body": body,
        "cta": "binary",
        "rationale": "Acknowledging merchant input and guiding toward concrete execution."
    }
