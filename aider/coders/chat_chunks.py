from dataclasses import dataclass, field
from typing import List


@dataclass
class ChatChunks:
    system: List = field(default_factory=list)
    examples: List = field(default_factory=list)
    done: List = field(default_factory=list)
    repo: List = field(default_factory=list)
    readonly_files: List = field(default_factory=list)
    chat_files: List = field(default_factory=list)
    cur: List = field(default_factory=list)
    reminder: List = field(default_factory=list)
    use_alternating_roles: bool = False

    def _ensure_alternating(self, messages):
        """Ensure messages alternate between user and assistant roles."""
        if not messages:
            return []
            
        result = []
        last_role = None
        
        for msg in messages:
            current_role = msg.get("role")
            if not current_role or current_role == "system":
                result.append(msg)
                continue
            
            if last_role == current_role:
                # Insert bridging message
                bridge_role = "assistant" if current_role == "user" else "user"
                result.append({
                    "role": bridge_role,
                    "content": "Continuing." if bridge_role == "assistant" else "Please proceed."
                })
            
            result.append(msg)
            last_role = current_role
            
        return result

    def all_messages(self):
        """Combine all chunks and ensure proper message alternation."""
        # First combine all messages
        all_msgs = (
            self.system
            + self.examples
            + self.readonly_files
            + self.repo
            + self.done
            + self.chat_files
            + self.cur
            + self.reminder
        )
        
        # Then ensure proper alternation
        if self.use_alternating_roles:
            return self._ensure_alternating(all_msgs)
        return all_msgs

    def add_cache_control_headers(self):
        if self.examples:
            self.add_cache_control(self.examples)
        else:
            self.add_cache_control(self.system)

        if self.repo:
            # this will mark both the readonly_files and repomap chunk as cacheable
            self.add_cache_control(self.repo)
        else:
            # otherwise, just cache readonly_files if there are any
            self.add_cache_control(self.readonly_files)

        self.add_cache_control(self.chat_files)

    def add_cache_control(self, messages):
        if not messages:
            return

        content = messages[-1]["content"]
        if type(content) is str:
            content = dict(
                type="text",
                text=content,
            )
        content["cache_control"] = {"type": "ephemeral"}

        messages[-1]["content"] = [content]

    def cacheable_messages(self):
        messages = self.all_messages()  # This now returns alternating messages
        for i, message in enumerate(reversed(messages)):
            if isinstance(message.get("content"), list) and message["content"][0].get(
                "cache_control"
            ):
                return messages[: len(messages) - i]
        return messages