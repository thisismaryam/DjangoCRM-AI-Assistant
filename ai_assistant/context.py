import json

from langchain_core.messages import BaseMessage, HumanMessage


MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARS = 6000


def trim_conversation_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep recent complete user turns within a bounded serialized size."""
    user_turn_starts = [
        index for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    if not user_turn_starts:
        return messages

    user_turn_starts = user_turn_starts[-MAX_HISTORY_TURNS:]
    history = messages[user_turn_starts[0]:]

    while (
        len(user_turn_starts) > 1
        and _serialized_size(history) > MAX_HISTORY_CHARS
    ):
        user_turn_starts = user_turn_starts[1:]
        history = messages[user_turn_starts[0]:]

    return history


def _serialized_size(messages: list[BaseMessage]) -> int:
    return sum(
        len(json.dumps(message.model_dump(mode="json", exclude_none=True), ensure_ascii=False))
        for message in messages
    )
