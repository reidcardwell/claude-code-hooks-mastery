# Git Workflow with Work Summary

Run complete git workflow then generate audio work completion summary.

## Steps

1. **Execute Git Workflow**: Run git-flow-automator agent to handle staging, commit, and push
2. **Generate Work Summary**: After git success, run enhanced-work-summary agent with git context
3. **Provide Audio Feedback**: Deliver cohesive audio summary about completed work

## Usage

```
/git-summary
```

## Implementation

Execute the following sequence:

### Step 1: Git Operations
Use the Task tool to run git-flow-automator:
```
Task(description="Execute git workflow", prompt="Update git", subagent_type="git-flow-automator")
```

### Step 2: Work Summary (Only if Step 1 succeeds)
After git operations complete successfully, use the Task tool to run enhanced-work-summary:
```
Task(description="Generate work completion summary", prompt="[Concise description of completed git work based on the commit message and changes]", subagent_type="enhanced-work-summary")
```

## Guidelines for Work Summary Prompt

When calling enhanced-work-summary, provide context based on the git-flow-automator results:
- Include specific files or features that were committed
- Mention the type of work (e.g., "Fixed configuration", "Added utility", "Updated documentation")
- Keep it factual and concise (1-2 sentences maximum)
- Focus on what was accomplished

## Example Prompts for enhanced-work-summary:
- "Committed sync-claude-config.sh utility for syncing .claude directories with backup functionality"
- "Fixed TTS configuration loops in hooks and updated git workflow integration"
- "Updated documentation with new API examples and troubleshooting guides"

## Notes

- This command combines both workflows into a single operation
- Git workflow must complete successfully before work summary is generated
- Maintains separation of concerns between the two agents
- Provides automatic audio feedback for completed git operations