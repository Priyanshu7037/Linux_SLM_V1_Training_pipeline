#!/usr/bin/env python3

"""
prepare_dataset.py

LinuxSLM Dataset Preparation
=============================

Reads all JSON files from:

    data/raw/

Supports:

    - JSON Array
    - JSON Object
    - JSON Lines (JSONL)

Extracts useful fields:

    - title
    - command
    - content
    - text
    - description
    - examples:
        - task
        - command

Output:

    data/dataset.txt

Each document is separated by:

    <eod>
"""

import json
import time
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/dataset.txt")

EOD = "<eod>"

# Approximate only.
# Real token count should be obtained using the LinuxSLM tokenizer.
CHARS_PER_TOKEN = 4


# ============================================================
# Global Statistics
# ============================================================

total_docs = 0
skipped_docs = 0
total_chars = 0

source_stats = {}


# ============================================================
# Text Cleaning
# ============================================================

def clean_text(value):
    """
    Clean text while preserving Linux/shell syntax.

    Important:
        We DO NOT lowercase text.

    Linux commands, paths, variables and arguments can be
    case-sensitive.
    """

    if not isinstance(value, str):
        return ""

    # Normalize line endings.
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    # Decode common HTML entities.
    value = value.replace("&lt;", "<")
    value = value.replace("&gt;", ">")
    value = value.replace("&amp;", "&")
    value = value.replace("&#39;", "'")
    value = value.replace("&quot;", '"')

    # Remove trailing whitespace from each line.
    lines = []

    for line in value.split("\n"):
        lines.append(line.rstrip())

    # Remove excessive consecutive blank lines.
    cleaned_lines = []

    blank_count = 0

    for line in lines:
        if line.strip():
            cleaned_lines.append(line)
            blank_count = 0
        else:
            blank_count += 1

            if blank_count <= 1:
                cleaned_lines.append("")

    value = "\n".join(cleaned_lines)

    return value.strip()


# ============================================================
# Add Structured Section
# ============================================================

def add_section(parts, label, value):
    """
    Add a labeled section if the value is not empty.
    """

    value = clean_text(value)

    if not value:
        return

    parts.append(f"[{label}]")
    parts.append(value)


# ============================================================
# Convert JSON Record → Training Text
# ============================================================

def record_to_text(record):
    """
    Convert one JSON record into structured training text.

    Supported:

        title
        command
        content
        text
        description
        examples[]
    """

    parts = []

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = record.get("title")

    if isinstance(title, str) and title.strip():
        add_section(
            parts,
            "TITLE",
            title
        )

    # --------------------------------------------------------
    # COMMAND
    # --------------------------------------------------------

    command = record.get("command")

    if isinstance(command, str) and command.strip():
        add_section(
            parts,
            "COMMAND",
            command
        )

    # --------------------------------------------------------
    # MAIN CONTENT
    # --------------------------------------------------------

    content_fields = (
        "content",
        "text",
        "description",
    )

    seen_content = set()

    for field in content_fields:
        value = record.get(field)

        if not isinstance(value, str):
            continue

        value = clean_text(value)

        if not value:
            continue

        # Prevent exact duplicate content.
        normalized = value.lower()

        if normalized in seen_content:
            continue

        seen_content.add(normalized)

        add_section(
            parts,
            "DESCRIPTION",
            value
        )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    examples = record.get("examples")

    if isinstance(examples, list):
        for example in examples:

            if not isinstance(example, dict):
                continue

            task = clean_text(
                example.get("task", "")
            )

            cmd = clean_text(
                example.get("command", "")
            )

            if not task and not cmd:
                continue

            parts.append("[EXAMPLE]")

            if task:
                parts.append(
                    f"Task: {task}"
                )

            if cmd:
                parts.append(
                    f"Command: {cmd}"
                )

    # --------------------------------------------------------
    # Final Text
    # --------------------------------------------------------

    return "\n\n".join(parts).strip()


# ============================================================
# Source Statistics
# ============================================================

def update_source_stats(path, written):
    """
    Track documents written/skipped for each source file.
    """

    name = path.name

    if name not in source_stats:
        source_stats[name] = {
            "written": 0,
            "skipped": 0,
        }

    if written:
        source_stats[name]["written"] += 1
    else:
        source_stats[name]["skipped"] += 1


# ============================================================
# Write One Record
# ============================================================

def write_record(record, output, source_path):
    """
    Convert and write one record.
    """

    global total_docs
    global skipped_docs
    global total_chars

    text = record_to_text(record)

    # Skip completely empty records.
    if not text:
        skipped_docs += 1

        update_source_stats(
            source_path,
            False
        )

        return

    # Write document.
    output.write(text)
    output.write("\n\n")

    # Document boundary.
    output.write(EOD)
    output.write("\n\n")

    # Statistics.
    total_docs += 1
    total_chars += len(text)

    update_source_stats(
        source_path,
        True
    )


# ============================================================
# Process JSON Array
# ============================================================

def process_json_array(path, output):
    """
    Process:

        [
            {...},
            {...},
            {...}
        ]
    """

    global skipped_docs

    try:
        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # Single object.
        if isinstance(data, dict):
            write_record(
                data,
                output,
                path
            )

        # Array of objects.
        elif isinstance(data, list):

            for record in data:

                if isinstance(record, dict):
                    write_record(
                        record,
                        output,
                        path
                    )

                else:
                    skipped_docs += 1

                    update_source_stats(
                        path,
                        False
                    )

    except Exception as e:
        print(
            f"[ERROR] {path.name}: {e}"
        )


# ============================================================
# Process JSONL
# ============================================================

def process_jsonl(path, output):
    """
    Process JSON Lines:

        {"title": "..."}
        {"title": "..."}
        {"title": "..."}
    """

    global skipped_docs

    try:
        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)

                    if isinstance(record, dict):
                        write_record(
                            record,
                            output,
                            path
                        )

                    else:
                        skipped_docs += 1

                        update_source_stats(
                            path,
                            False
                        )

                except json.JSONDecodeError:

                    skipped_docs += 1

                    update_source_stats(
                        path,
                        False
                    )

                    # Only print the first few malformed lines
                    # to avoid flooding the terminal.
                    if line_number <= 5:
                        print(
                            f"[WARNING] "
                            f"{path.name}: "
                            f"invalid JSON at line "
                            f"{line_number}"
                        )

    except Exception as e:
        print(
            f"[ERROR] {path.name}: {e}"
        )


# ============================================================
# Detect JSON File Type
# ============================================================

def process_file(path, output):
    """
    Detect whether the file is:

        JSON Array
        JSON Object
        JSONL
    """

    try:
        with path.open(
            "r",
            encoding="utf-8"
        ) as f:

            first = ""

            while True:
                ch = f.read(1)

                if not ch:
                    break

                if not ch.isspace():
                    first = ch
                    break

        # Empty file.
        if not first:
            print(
                f"[WARNING] Empty file: "
                f"{path.name}"
            )

            return

        # JSON array.
        if first == "[":
            process_json_array(
                path,
                output
            )

        # JSON object.
        elif first == "{":

            # A file beginning with { could be either:
            #
            #   {"foo": "..."}
            #
            # or JSONL:
            #
            #   {"foo": "..."}
            #   {"foo": "..."}
            #
            # Try normal JSON first.

            try:
                with path.open(
                    "r",
                    encoding="utf-8"
                ) as f:

                    data = json.load(f)

                if isinstance(
                    data,
                    dict
                ):
                    write_record(
                        data,
                        output,
                        path
                    )

                elif isinstance(
                    data,
                    list
                ):

                    for record in data:

                        if isinstance(
                            record,
                            dict
                        ):
                            write_record(
                                record,
                                output,
                                path
                            )

            except json.JSONDecodeError:

                # Not valid single JSON object.
                # Treat as JSONL.

                process_jsonl(
                    path,
                    output
                )

        # Otherwise assume JSONL.
        else:
            process_jsonl(
                path,
                output
            )

    except Exception as e:
        print(
            f"[ERROR] {path.name}: {e}"
        )


# ============================================================
# Print Source Statistics
# ============================================================

def print_source_statistics():

    print()
    print("=" * 80)
    print("Per-File Statistics")
    print("=" * 80)

    if not source_stats:
        print("No statistics available.")
        return

    for name in sorted(source_stats):

        stats = source_stats[name]

        written = stats["written"]
        skipped = stats["skipped"]

        print(
            f"{name:<50}"
            f"written={written:>8,} "
            f"skipped={skipped:>8,}"
        )


# ============================================================
# Main
# ============================================================

def main():

    start = time.time()

    # --------------------------------------------------------
    # Validate input directory
    # --------------------------------------------------------

    if not RAW_DIR.exists():

        print(
            f"ERROR: Directory not found: "
            f"{RAW_DIR}"
        )

        return

    # --------------------------------------------------------
    # Find JSON files
    # --------------------------------------------------------

    files = sorted(
        RAW_DIR.rglob("*.json")
    )

    if not files:

        print(
            "ERROR: No JSON files found."
        )

        return

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 80)
    print("LinuxSLM Dataset Preparation")
    print("=" * 80)

    print(
        f"Input directory : {RAW_DIR}"
    )

    print(
        f"Output file     : {OUTPUT_FILE}"
    )

    print(
        f"JSON files      : {len(files):,}"
    )

    print()

    # --------------------------------------------------------
    # Process everything
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output:

        for index, file in enumerate(
            files,
            start=1
        ):

            print(
                f"[{index:>3}/{len(files)}] "
                f"Processing {file.name}"
            )

            process_file(
                file,
                output
            )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    elapsed = time.time() - start

    estimated_tokens = (
        total_chars // CHARS_PER_TOKEN
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Dataset Preparation Complete")
    print("=" * 80)

    print(
        f"Output File          : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"JSON Files           : "
        f"{len(files):,}"
    )

    print(
        f"Documents Written    : "
        f"{total_docs:,}"
    )

    print(
        f"Skipped Documents    : "
        f"{skipped_docs:,}"
    )

    print(
        f"Characters           : "
        f"{total_chars:,}"
    )

    print(
        f"Estimated Tokens     : "
        f"{estimated_tokens:,}"
    )

    print(
        f"Estimated Tokens (M) : "
        f"{estimated_tokens / 1_000_000:.2f} M"
    )

    print(
        f"Time Taken           : "
        f"{elapsed:.2f} sec"
    )

    # --------------------------------------------------------
    # Source statistics
    # --------------------------------------------------------

    print_source_statistics()

    print()
    print("=" * 80)
    print("Ready for Tokenization / Training")
    print("=" * 80)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
