import os
import json
from typing import Dict, List, Any


def write_channel_messages_json(base_path: str, date_str: str, channel_name: str, messages: List[Dict[str, Any]]) -> str:
	"""Write the messages for a single channel to a JSON file.

	Returns the path to the written file.
	"""
	out_dir = os.path.join(base_path, "raw", "telegram_messages", date_str)
	os.makedirs(out_dir, exist_ok=True)

	safe_channel = channel_name.replace('/', '_')
	out_path = os.path.join(out_dir, f"{safe_channel}.json")

	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(messages, f, ensure_ascii=False, indent=2)

	return out_path


def write_manifest(base_path: str, date_str: str, channel_message_counts: Dict[str, int]) -> str:
	"""Write a manifest.json summarizing message counts per channel.

	Returns the path to the written manifest file.
	"""
	out_dir = os.path.join(base_path, "raw", "telegram_messages", date_str)
	os.makedirs(out_dir, exist_ok=True)

	manifest = {
		"date": date_str,
		"channel_message_counts": channel_message_counts,
	}

	out_path = os.path.join(out_dir, "manifest.json")
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(manifest, f, ensure_ascii=False, indent=2)

	return out_path


__all__ = ["write_channel_messages_json", "write_manifest"]

