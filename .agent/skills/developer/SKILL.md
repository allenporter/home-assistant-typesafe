---
name: project_development
description: Use standardized scripts to manage the project environment, testing, and linting.
---

# Project Development Skill

This skill teaches you how to interact with the codebase using standardized scripts, following the "Scripts to Rule Them All" pattern.

## Core Scripts

| Script                    | When to use                                                                    |
| :------------------------ | :----------------------------------------------------------------------------- |
| `script/bootstrap`        | After cloning or when dependencies change.                                     |
| `script/setup`            | Before starting to setup the virtual environment.                              |
| `script/test`             | Before submitting changes or to verify functionality.                          |
| `script/lint`             | Before committing to ensure code style and quality.                            |
| `script/update`           | When looking to sync down changes from the upstream repo.                      |
| `script/server`           | When you need a running server for integration testing or manual verification. |
| `script/sync-speculative` | Pulling speculative engine updates from a sibling integration.                 |

## Usage Patterns

### Standard Development Flow

1. **Initialize**: `./script/bootstrap`
2. **Pre-requistes**: `./script/setup` to start development
3. **Implement**: Make your changes to the code.
4. **Lint**: `./script/lint` to check for style issues.
5. **Test**: `./script/test` to run the test suite.
6. **Verify**: `./script/server` to run an ephemeral server for manual checks.

### Syncing the Speculative Engine (`script/sync-speculative`)

The `speculative/` package (`custom_components/<domain>/speculative`) and its synthetic unit tests (`tests/speculative/`) are shared across sibling integrations (e.g. Laya and Jev).

#### Rule: PULL ONLY (Never Push Across Repos)

Agents must **only pull** updates into their own repository. Never push changes into another integration's repository, as doing so can overwrite in-progress work by the user or another agent.

```bash
# CORRECT: Always run in YOUR repository, specifying the sibling repo as the source:
./script/sync-speculative /path/to/source-repo
```

#### What It Does

1. Auto-detects the domain names for both source and destination (e.g. `jev` source -> `laya` destination).
2. Copies `custom_components/<source_domain>/speculative/` -> `custom_components/<dest_domain>/speculative/` with `rsync --delete` (excluding `__pycache__` and `.pyc`).
3. Copies `tests/speculative/` -> destination's `tests/speculative/`.
4. Automatically rewrites test imports from `custom_components.<source_domain>` to `custom_components.<dest_domain>`.

#### Mandatory Diff Review & Verification Workflow

Immediately after syncing:

1. **Review Diff**: Run `git diff` and review every line changed. Verify that incoming changes make sense and do not inadvertently discard local improvements.
2. **Check Decoupling**: Confirm `custom_components/<domain>/speculative/` remains strictly self-contained with no leaking imports to parent `..const` or domain-specific modules.
3. **Lint & Test**: Run `./script/lint` and `./script/test` to ensure style checks and unit tests pass cleanly.

> [!CAUTION] > **Safety Guardrails:**
>
> - **Never push to sibling repos**: Only execute `sync-speculative` to pull changes into your current workspace (`./script/sync-speculative <source-repo>`).
> - **Check git status first**: Ensure your own working tree is clean before pulling.
> - **Never sync blindly**: Always inspect the resulting `git diff` before committing or pushing.

### Notes

- All scripts are located in the `script/` directory at the project root.
- Scripts are designed to be run from the project root.
- The scripts will automatically use `uv` if it is installed, otherwise they will fall back to standard Python tools.
