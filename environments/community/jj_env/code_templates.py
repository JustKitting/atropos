"""
Code Templates — Repo Content Factories
========================================

Small, parameterized code templates for building test repositories.
Each takes a seed for deterministic variation to prevent memorization.
"""

from __future__ import annotations

import random
import re as _re
from dataclasses import dataclass
from typing import Dict, List


# ── Target-finding utilities ─────────────────────────────────────────


@dataclass
class FunctionInfo:
    """A function definition found in Python source."""
    name: str
    start_line: int   # 0-indexed, line of 'def ...'
    end_line: int      # 0-indexed, last line of function body
    body_start: int    # 0-indexed, first line of body after 'def' line


@dataclass
class ReturnInfo:
    """A return statement found in Python source."""
    line_number: int   # 0-indexed
    line_text: str     # full text of the line (stripped)
    function_name: str # enclosing function name


@dataclass
class ConfigEntry:
    """A key-value pair found in YAML-like config."""
    key: str
    value: str
    line_number: int   # 0-indexed


def find_functions(content: str) -> List[FunctionInfo]:
    """Parse Python source for function definitions with line ranges."""
    lines = content.split("\n")
    functions: List[FunctionInfo] = []

    i = 0
    while i < len(lines):
        m = _re.match(r"^(def (\w+)\(.*\))", lines[i])
        if m:
            name = m.group(2)
            start = i
            body_start = i + 1
            # Walk forward to find end of function body
            j = i + 1
            while j < len(lines):
                line = lines[j]
                # Empty lines don't end functions
                if line.strip() == "":
                    j += 1
                    continue
                # Non-indented non-empty line = new top-level def or end
                if not line.startswith(" ") and not line.startswith("\t"):
                    break
                j += 1
            end = j - 1
            # Back up past trailing blank lines
            while end > start and lines[end].strip() == "":
                end -= 1
            functions.append(FunctionInfo(
                name=name, start_line=start, end_line=end, body_start=body_start,
            ))
            i = j
        else:
            i += 1

    return functions


def find_return_statements(content: str) -> List[ReturnInfo]:
    """Find return statements in Python source with enclosing function context."""
    lines = content.split("\n")
    results: List[ReturnInfo] = []
    current_func = "<module>"

    for i, line in enumerate(lines):
        m = _re.match(r"^def (\w+)\(", line)
        if m:
            current_func = m.group(1)
        stripped = line.strip()
        if stripped.startswith("return "):
            results.append(ReturnInfo(
                line_number=i,
                line_text=stripped,
                function_name=current_func,
            ))

    return results


def find_config_values(content: str) -> List[ConfigEntry]:
    """Parse YAML-like key-value pairs from config content."""
    lines = content.split("\n")
    results: List[ConfigEntry] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or stripped == "":
            continue
        if ":" in stripped and not stripped.endswith(":"):
            # Skip list items
            if stripped.startswith("- "):
                continue
            key, _, value = stripped.partition(":")
            value = value.strip().strip('"').strip("'")
            if value:
                results.append(ConfigEntry(key=key.strip(), value=value, line_number=i))

    return results


def python_calculator(seed: int) -> Dict[str, str]:
    """
    Generate a simple Python calculator module with seed-based variation.
    Returns dict of {filename: content}.
    """
    rng = random.Random(seed)

    func_names = {
        "add": rng.choice(["add", "sum_values", "plus"]),
        "subtract": rng.choice(["subtract", "minus", "sub"]),
        "multiply": rng.choice(["multiply", "mul", "product"]),
        "divide": rng.choice(["divide", "div", "quotient"]),
    }

    default_precision = rng.choice([2, 4, 6])
    module_doc = rng.choice([
        "Simple calculator module.",
        "Basic arithmetic operations.",
        "Math utility functions.",
    ])

    content = f'''"""
{module_doc}
"""


def {func_names["add"]}(a: float, b: float) -> float:
    """Return the sum of a and b."""
    return a + b


def {func_names["subtract"]}(a: float, b: float) -> float:
    """Return the difference of a and b."""
    return a - b


def {func_names["multiply"]}(a: float, b: float) -> float:
    """Return the product of a and b."""
    return a * b


def {func_names["divide"]}(a: float, b: float, precision: int = {default_precision}) -> float:
    """Return a divided by b, rounded to precision decimal places."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return round(a / b, precision)


def power(base: float, exp: float) -> float:
    """Return base raised to the power of exp."""
    return base ** exp


def modulo(a: int, b: int) -> int:
    """Return the remainder of a divided by b."""
    if b == 0:
        raise ValueError("Cannot modulo by zero")
    return a % b
'''

    return {"calc.py": content}


def python_utils(seed: int) -> Dict[str, str]:
    """
    Generate a Python utility module with string/data helpers.
    """
    rng = random.Random(seed)

    reverse_name = rng.choice(["reverse_string", "str_reverse", "flip_string"])
    count_name = rng.choice(["count_vowels", "vowel_count", "num_vowels"])
    validate_name = rng.choice(["is_valid_email", "validate_email", "check_email"])
    max_len = rng.choice([100, 200, 255])

    email_pattern = r'r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"'
    slugify_sub1 = r'r"[^\w\s-]"'
    slugify_sub2 = r'r"[\s_]+"'

    content = f'''"""
String and data processing utilities.
"""

import re


def {reverse_name}(s: str) -> str:
    """Reverse a string."""
    return s[::-1]


def {count_name}(s: str) -> int:
    """Count the number of vowels in a string."""
    return sum(1 for c in s.lower() if c in "aeiou")


def truncate(s: str, max_length: int = {max_len}) -> str:
    """Truncate string to max_length, adding ellipsis if needed."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


def {validate_name}(email: str) -> bool:
    """Check if a string looks like a valid email address."""
    pattern = {email_pattern}
    return bool(re.match(pattern, email))


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower().strip()
    text = re.sub({slugify_sub1}, "", text)
    text = re.sub({slugify_sub2}, "-", text)
    return text.strip("-")


def parse_csv_line(line: str, delimiter: str = ",") -> list:
    """Parse a single CSV line into a list of fields."""
    fields = []
    current = ""
    in_quotes = False
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == delimiter and not in_quotes:
            fields.append(current.strip())
            current = ""
        else:
            current += char
    fields.append(current.strip())
    return fields
'''

    return {"utils.py": content}


def config_yaml(seed: int) -> Dict[str, str]:
    """
    Generate a YAML configuration file with seed-based variation.
    """
    rng = random.Random(seed)

    db_host = rng.choice(["localhost", "db.internal", "10.0.0.5"])
    db_port = rng.choice([5432, 3306, 27017])
    db_name = rng.choice(["appdb", "production", "main_store"])
    log_level = rng.choice(["INFO", "DEBUG", "WARNING"])
    api_port = rng.choice([8080, 3000, 9000])
    cache_ttl = rng.choice([300, 600, 900])

    content = f"""# Application Configuration
app:
  name: myservice
  version: "1.0.0"
  environment: production

database:
  host: {db_host}
  port: {db_port}
  name: {db_name}
  pool_size: 10
  timeout: 30

logging:
  level: {log_level}
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: /var/log/app.log
  rotate: true
  max_size_mb: 50

api:
  host: 0.0.0.0
  port: {api_port}
  cors_origins:
    - "https://frontend.example.com"
    - "https://admin.example.com"
  rate_limit: 100
  timeout: 30

cache:
  backend: redis
  host: localhost
  port: 6379
  ttl: {cache_ttl}
  prefix: "app:"
"""

    return {"config.yaml": content}


def readme_md(seed: int) -> Dict[str, str]:
    """
    Generate a project README with seed-based variation.
    """
    rng = random.Random(seed)

    project_name = rng.choice(["DataPipeline", "WebService", "TaskRunner"])
    lang = rng.choice(["Python 3.11+", "Python 3.10+", "Python 3.12+"])
    license_type = rng.choice(["MIT", "Apache-2.0", "BSD-3-Clause"])

    content = f"""# {project_name}

A high-performance data processing service.

## Requirements

- {lang}
- PostgreSQL 15+
- Redis 7+

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and update the values.

## Usage

```bash
python -m {project_name.lower()} serve --port 8080
```

## Testing

```bash
pytest tests/ -v
```

## License

{license_type}
"""

    return {"README.md": content}


# Registry of all template functions
TEMPLATES = {
    "python_calculator": python_calculator,
    "python_utils": python_utils,
    "config_yaml": config_yaml,
    "readme_md": readme_md,
}


def generate_repo_files(
    template_names: list[str],
    seed: int,
) -> Dict[str, str]:
    """
    Generate files from multiple templates with a common seed.
    Returns combined dict of {filename: content}.
    """
    files = {}
    for i, name in enumerate(template_names):
        if name in TEMPLATES:
            files.update(TEMPLATES[name](seed + i))
    return files
