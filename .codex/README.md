# Codex Subagent Workflow

This directory contains configuration for parallel agent workflows in Codex.

## Configuration

- **config.toml**: Main configuration
  - `max_threads = 6`: Run up to 6 agents in parallel
  - `max_depth = 1`: Agents cannot spawn sub-agents

## Agents

### Explorer (`explorer.toml`)
Maps codebase structure and dependencies. Use for initial codebase understanding.

### Reviewer (`reviewer.toml`)
Reviews code for security and correctness. Use for quality assurance.

### Worker (`worker.toml`)
Implements features for specific modules. Spawn multiple workers for parallel development.

## Usage Examples

### Full Workflow
```
Run this as a subagent workflow. Spawn one explorer to map the code, one reviewer for security/correctness, and one worker per module. Wait for all agents, then summarize.
```

### Parallel Implementation
```
Spawn 3 worker agents: one for the data processing module, one for the API layer, and one for the UI components. Coordinate their work.
```

### Code Review Workflow
```
Spawn one explorer to understand the changes, then one reviewer to check security and correctness. Report findings.
```

## Workflow Patterns

1. **Explore → Review → Implement**
   - Explorer maps the codebase
   - Reviewer identifies issues
   - Workers implement fixes in parallel

2. **Parallel Development**
   - Multiple workers on different modules
   - Coordinator agent synthesizes results

3. **Quality Gate**
   - Explorer + Reviewer before any implementation
   - Ensures understanding before changes
